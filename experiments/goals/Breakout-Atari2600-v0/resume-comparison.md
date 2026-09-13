# Matched uninterrupted versus resumed training

Compare continued learning with an ordinary checkpoint restart using a common
training prefix. A shared seed alone is insufficient: the resumed arm must load
an actual branch checkpoint emitted by the uninterrupted arm.

For each pair, keep the policy, Adam state, environment/reward contract, absolute
learning-rate milestones, total endpoint, update count after branching, source,
runtime and evaluation episodes matched. Let the uninterrupted arm continue
through the branch without resetting its environment or random streams. Start
the resumed arm from that branch using the normal trainer startup, which resets
environments and random streams. This contrast measures the combined ordinary
restart effect, not each reset component independently.

The custom PPO backend accepts `checkpoint_update_steps`, an ordered list of
absolute transition counts at which to save after completing the PPO update.
Use `train.checkpoint_freq=0` with this option to avoid overlapping transition
checkpoints. Future branch points must align with rollout boundaries relative
to the loaded model step (`n_envs * n_steps`). Points at or before the loaded step
are already passed; the final checkpoint covers the terminal endpoint. Branch
at a completed update so a restart does not discard an unfinished rollout or
silently omit its optimizer update. Default periodic checkpoint behavior is
unchanged when the option is empty.

Freeze the selected parent, branch steps, schedule, paired training seeds,
evaluation seeds and resource limits before launching. Verify the branch bundle
hash and retained optimizer state, and compare the actual learning-rate values
and completed updates. Evaluate all declared episodes of the frozen branch,
intermediate and final policies. Record failures and do not select replacement
branches after observing results. Report paired results as research evidence;
this experiment alone does not prove a recipe trains from scratch or achieves
untouched held-out acceptance.
