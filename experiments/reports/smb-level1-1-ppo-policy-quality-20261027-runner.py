"""Run one approved exhaustive, non-Acceptance assessment in the frozen image."""
from pathlib import Path
from datetime import datetime, timezone
import copy, hashlib, json, os, sys, time
import torch
from gradlab.model_sources import download_public_checkpoint_manifest_source
from gradlab.policy_bundle import evaluation_contract
from gradlab.eval_runner import evaluate_policy_bundle
from gradlab.checkpoint_acceptance import validate_episode_rows, acceptance_aggregates
from gradlab.video import SingleEpisodeVideoRecorder
from gradlab.rom_runtime import bind_cached_rom

ROOT = Path('/assessment')
OUT = Path('/output')
plan = json.loads((ROOT / 'plan.json').read_text())
row = next(r for r in plan['selected'] if r['label'] == sys.argv[1])
target = OUT / row['label']
target.mkdir(exist_ok=True)
assert not (target / 'result.json').exists(), 'Assessment already has a result'
started = datetime.now(timezone.utc).isoformat()
start = time.monotonic()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
source = download_public_checkpoint_manifest_source(row['manifest_url'], root=target / 'bundle')
bundle = source.bundle
assert bundle.checkpoint_sha256 == row['checkpoint']['sha256']
assert evaluation_contract(bundle.recipe) == plan['base_evaluation_contract']
contract = copy.deepcopy(plan['assessment_contract'])
assert contract['evidence_policy']['fail_fast'] == 'disabled'
video = SingleEpisodeVideoRecorder(target / 'episode.mp4', episode_id='lane-00-episode-000', fps=30, max_bytes=256 * 1024**2)
summary, _ = evaluate_policy_bundle(
    bundle, device='cpu', episodes=100, n_envs=16, watchdog_steps=4500,
    progress=True, acceptance_contract=contract, episode_video_capture=video,
    rom_binding=bind_cached_rom(contract['asset'], cache_root=Path('/rom-cache')),
    internal_execution_id=f"{plan['study']}:{row['checkpoint']['checkpoint_id']}",
)
episodes = validate_episode_rows(summary['episode_results'], contract=contract)
assert len(episodes) == 100
aggregates = acceptance_aggregates(episodes, contract=contract)
video_meta = video.require_complete()
video_path = target / 'episode.mp4'
assert video_path.is_file()
result = {
    'schema_version': 1, 'assessment_id': f"{plan['study']}:{row['checkpoint']['checkpoint_id']}",
    'protocol': plan['protocol'], 'acceptance_evidence': False, 'promotion': None,
    'run_id': row['run_id'], 'checkpoint_id': row['checkpoint']['checkpoint_id'],
    'checkpoint_sha256': bundle.checkpoint_sha256, 'recipe_sha256': bundle.recipe_sha256,
    'source_sha': plan['source_sha'], 'runtime_image': plan['runtime_image'],
    'assessment_contract': contract, 'started_at': started,
    'completed_at': datetime.now(timezone.utc).isoformat(), 'duration_seconds': time.monotonic() - start,
    'episode_results': episodes, 'aggregates': aggregates,
    'evaluation_evidence': summary['evaluation_evidence'],
    'representative_video': {**video_meta, 'sha256': hashlib.sha256(video_path.read_bytes()).hexdigest(), 'size_bytes': video_path.stat().st_size},
}
encoded = json.dumps(result, sort_keys=True, indent=2, default=lambda x: x.item()) + '\n'
temporary = target / 'result.partial'
temporary.write_text(encoded)
temporary.replace(target / 'result.json')
print(json.dumps({'label': row['label'], 'episodes': len(episodes), 'successes': aggregates['success_count'], 'return': aggregates.get('eval/return/mean'), 'duration_seconds': result['duration_seconds']}), flush=True)
