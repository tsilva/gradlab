"""Bounded lockstep diagnostic of unmodified SB3 and GradLab PPO operations."""
from __future__ import annotations
import argparse, copy, hashlib, importlib.metadata, inspect, json, os, platform, random, signal, sys, threading, time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed, obs_as_tensor
from gradlab.ppo import GradLabPPO
from gradlab.env import resolve_env_config, make_vec_envs
from gradlab.env_metadata import env_config_from_config_dict, ENV_CONFIG_METADATA_KEYS
from gradlab.rom_runtime import bind_cached_rom, runtime_cache_root
from gradlab.training.sb3_on_policy import policy_type_for_config, policy_kwargs_from_config
from gradlab.training.ppo_engine import TensorRolloutBuffer, _CompiledPolicyCalls, _Precision, _ppo_update, _configure_optimizer_for_device
from gradlab.schedules import learning_rate_schedule, scheduled_scalar
from gradlab.training.sb3 import _normalize_ppo

parser=argparse.ArgumentParser()
parser.add_argument('--plan',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
parser.add_argument('--steps',type=int,default=512)
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--deterministic',action='store_true')
parser.add_argument('--shared-targets',action='store_true')
args=parser.parse_args();PLAN=json.loads(args.plan.read_text());OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'result.json').exists(), 'Preserve completed diagnostic; select a distinct output directory'
START=time.monotonic();device=torch.device(args.device);torch.set_num_threads(1);torch.set_num_interop_threads(1)
if args.deterministic:
    if device.type=='cuda':assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') in {':4096:8',':16:8'}
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Diagnostic finite deadline reached')))
signal.alarm(PLAN['max_seconds'])
REPORT={'diagnostic_id':PLAN['diagnostic_id'],'source_sha':PLAN['source_sha'],'device':str(device),'platform':platform.platform(),'steps':args.steps,'n_envs':16,'epochs':args.epochs,'shared_targets':args.shared_targets,'deterministic_requested':args.deterministic,'versions':{x:importlib.metadata.version(x) for x in ['torch','numpy','stable-baselines3','env-supermariobrosnes-turbo-emu']},'stages':[],'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'first_exact_difference':None,'first_beyond_tolerance':None,'representation_differences':[]}

def tensor(v):
    if isinstance(v,torch.Tensor):return v.detach()
    return torch.as_tensor(v)

def digest(v):
    a=tensor(v).cpu().contiguous().numpy();return hashlib.sha256(a.tobytes()).hexdigest()

def compare(stage,left,right,**where):
    assert left.keys()==right.keys(),f'Field mismatch at {stage}'
    differences=[]
    for name in left:
        a=tensor(left[name]).cpu();b=tensor(right[name]).cpu();assert a.shape==b.shape,(stage,name,a.shape,b.shape)
        if a.dtype!=b.dtype:
            entry={'stage':stage,'field':name,'left_dtype':str(a.dtype),'right_dtype':str(b.dtype)}
            if entry not in REPORT['representation_differences']:REPORT['representation_differences'].append(entry)
        if torch.equal(a,b):continue
        aa=a.double();bb=b.double();diff=(aa-bb).abs();finite=torch.isfinite(diff);maxabs=float(diff[finite].max().cpu()) if finite.any() else None
        same_tol=bool(torch.allclose(aa,bb,rtol=2e-6,atol=2e-6,equal_nan=True))
        item={'stage':stage,'field':name,**where,'shape':list(a.shape),'left_dtype':str(a.dtype),'right_dtype':str(b.dtype),'max_abs':maxabs,'different_values':int(torch.count_nonzero(a!=b).cpu()),'within_2e_6':same_tol,'left_sha256':digest(a),'right_sha256':digest(b)}
        differences.append(item)
        if REPORT['first_exact_difference'] is None:
            REPORT['first_exact_difference']=item;print(json.dumps({'event':'first_exact_difference',**item}),flush=True)
        if not same_tol and REPORT['first_beyond_tolerance'] is None:REPORT['first_beyond_tolerance']=item
    REPORT['stages'].append({'stage':stage,**where,'fields':len(left),'different_fields':len(differences),'differences':differences[:3]})
    return not differences

def rng_state():
    if device.type=='cuda':torch.cuda.synchronize()
    return {'python':copy.deepcopy(random.getstate()),'numpy':copy.deepcopy(np.random.get_state()),'torch':torch.get_rng_state().clone(),'cuda':[x.clone() for x in torch.cuda.get_rng_state_all()] if device.type=='cuda' else []}

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])

def rng_hash(s):
    h=hashlib.sha256(repr(s['python']).encode());h.update(s['numpy'][1].tobytes());h.update(repr((s['numpy'][0],s['numpy'][2:])).encode());h.update(s['torch'].numpy().tobytes())
    for x in s['cuda']:h.update(x.cpu().numpy().tobytes())
    return h.hexdigest()

class FirstActionDifference(Exception):pass

def named_weights(model):return dict(model.policy.named_parameters())
def gradients(model):return {k:v.grad for k,v in model.policy.named_parameters() if v.grad is not None}
def optimizer_tensors(model):
    out={}
    for name,p in model.policy.named_parameters():
        for k,v in model.policy.optimizer.state.get(p,{}).items():
            if torch.is_tensor(v):out[name+'/'+k]=v
    return out

cfg=PLAN['train_config'];backend=_normalize_ppo(cfg['training_backend']['config'],label='diagnostic.ppo');common={**cfg,'seed':PLAN['seed']};n_envs=16
config=resolve_env_config(env_config_from_config_dict({k:v for k,v in cfg.items() if k in ENV_CONFIG_METADATA_KEYS}))
env=make_vec_envs(config,n_envs=n_envs,seed=PLAN['seed'],rom_binding=bind_cached_rom(PLAN['asset'],cache_root=runtime_cache_root()),native_kwargs_overrides={'num_threads':1})
models=[]
try:
    kwargs={'policy':policy_type_for_config(env.observation_space,common),'env':env,'learning_rate':learning_rate_schedule(common,backend),'n_steps':args.steps,'batch_size':512,'n_epochs':args.epochs,'gamma':backend['gamma'],'gae_lambda':backend['gae_lambda'],'ent_coef':backend['ent_coef'],'vf_coef':backend['vf_coef'],'clip_range':backend['clip_range'],'clip_range_vf':backend['clip_range_vf'],'normalize_advantage':False,'target_kl':backend['target_kl'],'policy_kwargs':policy_kwargs_from_config(backend,common_config=common,optimizer_eps=backend['adam_eps']),'device':str(device),'verbose':0}
    set_random_seed(PLAN['seed']);reference=PPO(**kwargs);models.append(reference)
    initial_rng=rng_state();candidate=GradLabPPO(**kwargs);models.append(candidate)
    candidate.policy.load_state_dict(copy.deepcopy(reference.policy.state_dict()));candidate.policy.optimizer.load_state_dict(copy.deepcopy(reference.policy.optimizer.state_dict()));candidate.rollout_buffer=None
    _configure_optimizer_for_device(candidate.policy.optimizer,device,fused=False)
    for model in models:model.set_logger(configure(format_strings=[]));model.policy.set_training_mode(False)
    REPORT['optimizer_configuration']={role:{'defaults':{k:v for k,v in m.policy.optimizer.defaults.items() if k!='params'},'groups':[{k:v for k,v in g.items() if k!='params'} for g in m.policy.optimizer.param_groups]} for role,m in zip(['reference','candidate'],models)}
    restore_rng(initial_rng);compare('initial_weights',named_weights(reference),named_weights(candidate));compare('initial_optimizer',optimizer_tensors(reference),optimizer_tensors(candidate))
    REPORT['initial_parameter_sha256']={k:digest(v) for k,v in named_weights(reference).items()};REPORT['initial_rng_sha256']=rng_hash(initial_rng)
    observations=env.reset();REPORT['initial_observation_sha256']=digest(observations);REPORT['observation_shape']=list(observations.shape);REPORT['action_space']=str(env.action_space)
    buffer=TensorRolloutBuffer.allocate(observations,action_space=env.action_space,n_steps=args.steps,n_envs=n_envs,device=device)
    starts=np.ones(n_envs,dtype=bool);last_dones=np.zeros(n_envs,dtype=bool);logits={}
    hooks=[m.policy.action_net.register_forward_hook(lambda _m,_a,out,role=role:logits.__setitem__(role,out.detach().clone())) for m,role in zip(models,['reference','candidate'])]
    for step in range(args.steps):
        obs=np.array(observations,copy=True);ct=buffer.begin_step(obs,torch.as_tensor(starts,device=device));state=rng_state()
        if step==0:REPORT['first_action_rng_sha256']=rng_hash(state)
        with torch.no_grad():a,v,lp=reference.policy(obs_as_tensor(obs,device))
        after=rng_state();restore_rng(state)
        with torch.no_grad():ca,cv,clp=candidate.policy(ct)
        candidate_after=rng_state();assert rng_hash(after)==rng_hash(candidate_after),'Action sampling RNG consumption differs';restore_rng(after)
        compare('rollout_forward',{'logits':logits['reference'],'actions':a,'values':v,'log_prob':lp},{'logits':logits['candidate'],'actions':ca,'values':cv,'log_prob':clp},rollout_step=step)
        if not torch.equal(a,ca):raise FirstActionDifference('Action mismatch; stop before a hybrid trajectory can contaminate later stages')
        observations,rewards,dones,infos=env.step(a.cpu().numpy());r_ref=np.array(rewards,copy=True);original=np.array(rewards,copy=True)
        truncated=np.asarray([bool(done and info.get('TimeLimit.truncated',False)) for done,info in zip(dones,infos)]);lanes=np.flatnonzero(truncated);terminal_vals=[]
        if lanes.size:
            with torch.no_grad():
                for lane in lanes:
                    terminal=reference.policy.obs_to_tensor(infos[lane]['terminal_observation'])[0];value=reference.policy.predict_values(terminal)[0];terminal_vals.append(value.flatten()[0]);r_ref[lane]+=float(reference.gamma)*value
                terminal_batch=np.stack([infos[lane]['terminal_observation'] for lane in lanes]);batched=candidate.policy.predict_values(obs_as_tensor(terminal_batch,device)).flatten()
            compare('time_limit_values',{'values':torch.stack(terminal_vals)},{'values':batched},rollout_step=step,lanes=lanes.tolist())
        reference.rollout_buffer.add(obs,a.cpu().numpy().reshape(n_envs,1),r_ref,starts,v,lp)
        slot=buffer.end_step(ca,original,cv,clp)
        if lanes.size:slot[torch.as_tensor(lanes,device=device)]+=float(candidate.gamma)*batched
        compare('rollout_rewards',{'rewards':r_ref},{'rewards':slot},rollout_step=step)
        starts=np.asarray(dones,dtype=bool).copy();last_dones=starts
        if (step+1)%64==0:print(json.dumps({'event':'rollout_progress','steps':step+1,'elapsed':time.monotonic()-START}),flush=True)
    for h in hooks:h.remove()
    with torch.no_grad():last=reference.policy.predict_values(obs_as_tensor(observations,device));clast=candidate.policy.predict_values(obs_as_tensor(observations,device))
    compare('last_values',{'values':last},{'values':clast})
    def gae_trace(function,kw):
        code=function.__func__.__code__;lines,line0=inspect.getsourcelines(function);target=next(line0+i for i,line in enumerate(lines) if 'self.advantages[step]' in line);records=[]
        def trace(frame,event,arg):
            if frame.f_code is code and event=='line' and frame.f_lineno==target:
                q=frame.f_locals;records.append({'step':q['step'],'delta':tensor(q['delta']).detach().cpu().clone(),'gae':tensor(q.get('last_gae_lam',q.get('last_gae'))).detach().cpu().clone()})
            return trace if frame.f_code is code else None
        sys.settrace(trace)
        try:function(**kw)
        finally:sys.settrace(None)
        return records
    ra=gae_trace(reference.rollout_buffer.compute_returns_and_advantage,{'last_values':last,'dones':last_dones})
    rb=gae_trace(buffer.finish,{'last_values':clast,'dones':torch.as_tensor(last_dones,device=device),'gamma':backend['gamma'],'gae_lambda':backend['gae_lambda']})
    assert len(ra)==len(rb)==args.steps
    for x,y in zip(ra,rb):assert x['step']==y['step'];compare('gae_recurrence',{'delta':x['delta'],'gae':x['gae']},{'delta':y['delta'],'gae':y['gae']},reverse_step=x['step'])
    compare('gae_final',{'advantages':reference.rollout_buffer.advantages,'returns':reference.rollout_buffer.returns},{'advantages':buffer.advantages,'returns':buffer.returns})
    if args.shared_targets:
        buffer.advantages.copy_(torch.as_tensor(reference.rollout_buffer.advantages,device=device));buffer.returns.copy_(torch.as_tensor(reference.rollout_buffer.returns,device=device))
    # Each production loop sees the same private NumPy RNG snapshot; the harness records every shuffle.
    base_rng=rng_state();local=threading.local();condition=threading.Condition();turn='reference';waiting={};failure=[];event_count=0
    role_models={'reference':reference,'candidate':candidate};role_state={x:{'batch':0,'optimizer_steps':0,'epoch':-1,'rng':np.random.RandomState()} for x in role_models}
    for q in role_state.values():q['rng'].set_state(copy.deepcopy(base_rng['numpy']))
    def checkpoint(stage,data):
        nonlocal_dummy=None
        global turn,event_count
        role=local.role
        with condition:
            if failure:raise RuntimeError('Peer diagnostic failed')
            if role=='reference':
                waiting.clear();waiting.update(stage=stage,data=data,batch=role_state[role]['batch']);turn='candidate';condition.notify_all()
                while turn!='reference' and not failure:condition.wait(.5)
            else:
                if waiting.get('stage')!=stage:
                    failure.append({'left_stage':waiting.get('stage'),'right_stage':stage});condition.notify_all();raise RuntimeError('Production loops took different control-flow paths')
                compare('update/'+stage,waiting['data'],data,minibatch=role_state[role]['batch']);event_count+=1
                if stage=='optimizer_after' and role_state[role]['batch']%8==0:print(json.dumps({'event':'update_progress','minibatch':role_state[role]['batch'],'elapsed':time.monotonic()-START}),flush=True)
                turn='reference';condition.notify_all()
                if stage!='end':
                    while turn!='candidate' and not failure:condition.wait(.5)
            if failure:raise RuntimeError('Peer diagnostic failed')
    original_perm=np.random.permutation;original_clip=torch.nn.utils.clip_grad_norm_;original_backward=torch.Tensor.backward
    def permutation(n):
        if not hasattr(local,'role'):return original_perm(n)
        q=role_state[local.role];q['epoch']+=1;q['permutation']=q['rng'].permutation(n);q['within_epoch']=0;checkpoint('shuffle',{'indices':q['permutation']});return q['permutation'].copy()
    def flat(v):return tensor(v).to(device).transpose(0,1).flatten(0,1)
    buffers={'reference':{'old_values':flat(reference.rollout_buffer.values),'old_log_probs':flat(reference.rollout_buffer.log_probs),'advantages':flat(reference.rollout_buffer.advantages),'returns':flat(reference.rollout_buffer.returns)},'candidate':{'old_values':flat(buffer.values),'old_log_probs':flat(buffer.log_probs),'advantages':flat(buffer.advantages),'returns':flat(buffer.returns)}}
    originals=[];raw_logits={}
    for role,model in role_models.items():
        original_eval=model.policy.evaluate_actions;original_step=model.policy.optimizer.step
        def evaluate(obs,actions,*,role=role,orig=original_eval):
            q=role_state[role];q['batch']+=1;idx=q['permutation'][q['within_epoch']:q['within_epoch']+512];q['within_epoch']+=512;q['old']={k:v[torch.as_tensor(idx,device=v.device)] for k,v in buffers[role].items()}
            checkpoint('inputs',{'observations':obs,'actions':actions,**q['old']});values,lp,entropy=orig(obs,actions);q['outputs']=(values.flatten(),lp,entropy);checkpoint('forward',{'logits':raw_logits[role],'values':values,'log_prob':lp,'entropy':entropy})
            with torch.no_grad():kl=((torch.exp(lp-q['old']['old_log_probs'])-1)-(lp-q['old']['old_log_probs'])).mean()
            threshold=float('inf') if role_models[role].target_kl is None else 1.5*role_models[role].target_kl
            checkpoint('kl_guard',{'approx_kl':kl,'threshold':threshold,'would_stop':bool(kl>threshold)})
            return values,lp,entropy
        def opt_step(*aa,role=role,model=model,orig=original_step,**kw):
            checkpoint('optimizer_before',{'weight/'+k:v for k,v in named_weights(model).items()}|{'adam/'+k:v for k,v in optimizer_tensors(model).items()});value=orig(*aa,**kw);role_state[role]['optimizer_steps']+=1;checkpoint('optimizer_after',{'weight/'+k:v for k,v in named_weights(model).items()}|{'adam/'+k:v for k,v in optimizer_tensors(model).items()});return value
        originals.append((model,original_eval,original_step));model.policy.evaluate_actions=evaluate;model.policy.optimizer.step=opt_step
        hooks.append(model.policy.action_net.register_forward_hook(lambda _m,_a,out,role=role:raw_logits.__setitem__(role,out)))
    def backward(loss,*aa,**kw):
        if hasattr(local,'role'):
            q=role_state[local.role];v,lp,entropy=q['outputs'];old=q['old']
            with torch.no_grad():
                ratio=torch.exp(lp-old['old_log_probs']);pl=-torch.min(old['advantages']*ratio,old['advantages']*torch.clamp(ratio,.85,1.15)).mean();vl=torch.nn.functional.mse_loss(old['returns'],v);el=-entropy.mean();kl=((torch.exp(lp-old['old_log_probs'])-1)-(lp-old['old_log_probs'])).mean()
            checkpoint('loss',{'actual_total_loss':loss,'policy_loss':pl,'value_loss':vl,'entropy_loss':el,'approx_kl':kl})
        return original_backward(loss,*aa,**kw)
    def clip(parameters,*aa,**kw):
        if not hasattr(local,'role'):return original_clip(parameters,*aa,**kw)
        model=role_models[local.role];checkpoint('raw_gradients',gradients(model));norm=original_clip(parameters,*aa,**kw);checkpoint('clipped_gradients',gradients(model)|{'total_norm':norm});return norm
    np.random.permutation=permutation;torch.Tensor.backward=backward;torch.nn.utils.clip_grad_norm_=clip
    progress=max(1-args.steps*n_envs/common['timesteps'],0);ent=scheduled_scalar(backend['ent_coef'],backend['ent_coef_final'],args.steps*n_envs,backend['ent_coef_schedule_timesteps'])
    for model in models:model._current_progress_remaining=progress;model.ent_coef=ent
    def worker(role):
        global turn
        local.role=role
        try:
            with condition:
                while turn!=role and not failure:condition.wait(.5)
            if role=='reference':reference.train()
            else:_ppo_update(candidate,buffer,calls=_CompiledPolicyCalls(candidate.policy,device,compile_policy=False),precision=_Precision('fp32',device),progress_remaining=progress,normalization_mode='disabled',advantage_context=None,ent_coef=ent,torch_permutation=False)
            checkpoint('end',{'updates':role_models[role]._n_updates,'evaluated_minibatches':role_state[role]['batch'],'optimizer_steps':role_state[role]['optimizer_steps']})
        except BaseException as exc:
            with condition:failure.append({'role':role,'type':type(exc).__name__,'message':str(exc)});condition.notify_all()
    threads=[threading.Thread(target=worker,args=(role,),daemon=True) for role in role_models]
    try:
        for t in threads:t.start()
        while any(t.is_alive() for t in threads):
            for t in threads:t.join(.2)
            if failure:raise RuntimeError(str(failure))
    finally:
        if any(t.is_alive() for t in threads):
            with condition:failure.append({'reason':'coordinator_abort'});condition.notify_all()
            for t in threads:t.join(2)
        np.random.permutation=original_perm;torch.Tensor.backward=original_backward;torch.nn.utils.clip_grad_norm_=original_clip
        for model,ev,st in originals:model.policy.evaluate_actions=ev;model.policy.optimizer.step=st
        for h in hooks:
            try:h.remove()
            except Exception:pass
    assert rng_hash(rng_state())==rng_hash(base_rng),'Unexpected global RNG consumption during deterministic update fixture'
    assert all(np.array_equal(role_state['reference']['rng'].get_state()[1],role_state['candidate']['rng'].get_state()[1]) for _ in [0])
    compare('final_weights',named_weights(reference),named_weights(candidate));compare('final_adam',optimizer_tensors(reference),optimizer_tensors(candidate))
    REPORT.update(status='complete',elapsed_seconds=time.monotonic()-START,update_stage_pairs=event_count,reference_minibatches=role_state['reference']['batch'],candidate_minibatches=role_state['candidate']['batch'],reference_optimizer_steps=role_state['reference']['optimizer_steps'],candidate_optimizer_steps=role_state['candidate']['optimizer_steps'],torch_rng_unchanged_during_update=True)
except FirstActionDifference as exc:
    REPORT.update(status='complete_first_divergence',stopped_at='rollout_action',reason=str(exc),elapsed_seconds=time.monotonic()-START)
except BaseException as exc:
    REPORT.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)},elapsed_seconds=time.monotonic()-START)
    raise
finally:
    env.close();signal.alarm(0);(OUT/'result.json').write_text(json.dumps(REPORT,indent=2)+'\n');print(json.dumps({'event':'diagnostic_terminal','status':REPORT['status'],'first_exact_difference':REPORT['first_exact_difference'],'first_beyond_tolerance':REPORT['first_beyond_tolerance'],'elapsed_seconds':REPORT['elapsed_seconds']}),flush=True)
