# SMB Level1-1 PPO execution profiles: stopped early

The user stopped this study on 2026-10-28 and selected **custom `gradlab.ppo`, FP32, `sb3-parity` (uncompiled, unfused Adam, NumPy shuffle)** as the Level1-1 recipe default. This is an operational choice supported by partial evidence, not a completed five-seed superiority result.

Thirteen learners completed their 8,003,584-transition budget. Twelve have all eight verified 400-episode assessments: **97 checkpoints / 38,800 episodes**. The thirteenth learner (max-throughput, seed 300) finished and has one verified 400-episode assessment (259 clears at 1,007,616 steps). Its original assessment launch was already in flight when cancellation was requested; the container appeared after the first absence check and completed successfully. The same original result/video were collected and verified, and the container was removed without recomputation. Its remaining seven checkpoint assessments were canceled. Twelve planned learners were never admitted. Remaining work and the queued study were canceled; no automatic continuation remains.

The frozen source is `2069964e7ce38de8f87392de084145e6af8f87d3`; runtime digest is `sha256:e5b0b2978a608984c6d499d272de11081c6b8b07a7c9354a9bdcc865071cc1a3`; plan SHA256 is `dd0d9f84f3466544312e693793b394bbe49c1006af5a34e6512cdbc43e65c74c`. The working recipe change did not alter these inputs.

## Measured results

| Profile | Seed | First measured ≥95% (M steps) | Final clears | Quality AUC | Weighted steps/s | Learner minutes |
|---|---:|---:|---:|---:|---:|---:|
| SB3 | 100 | 5.005 | 99.75% | 0.912 | 6962 | 19.28 |
| Custom uncompiled/unfused | 100 | 3.006 | 99.50% | 0.891 | 7621 | 17.69 |
| Custom compiled | 100 | 4.006 | 96.50% | 0.834 | 7469 | 18.35 |
| Custom compiled/fused | 100 | 6.005 | 99.75% | 0.847 | 7667 | 17.93 |
| Custom max-throughput | 100 | 3.006 | 98.50% | 0.890 | 7668 | 17.93 |
| Custom uncompiled/unfused | 200 | 4.006 | 94.75% | 0.880 | 7636 | 17.60 |
| Custom compiled | 200 | 7.004 | 95.75% | 0.837 | 7508 | 18.31 |
| Custom compiled/fused | 200 | 3.006 | 99.25% | 0.884 | 7676 | 17.93 |
| Custom max-throughput | 200 | 3.006 | 96.00% | 0.910 | 7694 | 17.86 |
| SB3 | 200 | 4.006 | 79.25% | 0.831 | 7045 | 19.19 |
| Custom compiled | 300 | 5.005 | 100.00% | 0.828 | 7523 | 18.24 |
| Custom compiled/fused | 300 | 5.005 | 94.50% | 0.868 | 7655 | 17.91 |

Weighted throughput uses the common 1M–4M window. Learner time includes initialization and compilation. Quality AUC is normalized clear-rate area over the common assessed checkpoint grid. The threshold is an empirical 380/400 clears; crossings are resolved only to adjacent checkpoints and can be transient. Missing assessments are unavailable evidence, not failed episodes or measured threshold nonattainment.

## Interpretation and default decision

- In the two fully paired seeds, uncompiled/unfused custom PPO ran 8.4–9.5% faster than SB3 over the common throughput window. It also finished the full learner budget sooner.
- Compilation alone ran roughly 2% slower than uncompiled custom PPO in both fully paired seeds, and had lower full-curve quality AUC in both. Startup cost alone does not explain that throughput difference.
- Max-throughput was about 9–10% faster than SB3 and first met the quality threshold at the 3M checkpoint in both complete paired seeds. That arm bundles compilation, fusion, shuffle/RNG and layout differences; this study does not identify an isolated mechanism.
- Quality is unstable in multiple arms: SB3 seed 200 fell from 99% at 7M to 79.25% at 8M; compiled custom seed 300 moved from 95.75% at 5M to 39% at 6M, then recovered to 100% at 8M. Compiled/fused seed 300 ended at 94.5%. Neither one first crossing nor one final score describes the whole trajectory.
- Keep uncompiled/unfused custom PPO as the user-selected practical default for this environment. The remaining seeds were not completed; no robust claim of universal sample-efficiency or final-quality superiority follows. Fusion without compilation was not isolated by this study.

## Limits and retained evidence

Training seeds are the independent units. The 400 episodes measure conditional policy quality; episode confidence intervals do not estimate uncertainty across training seeds. The same assessment panel was reused across checkpoints. Prior historical cohorts are separate and are not pooled here. No assessments grant Acceptance, Promotion or Release.

`sb3-parity` names limited execution controls, not proven end-to-end parity. Residual differences include CPU NumPy versus device Torch advantage calculation and per-lane versus batched timeout bootstrapping. Original training runs did not capture step-zero parameter/RNG hashes or enforce deterministic CUDA/TF32 policies. The cause of historical trajectory divergence remains unknown.

The separate controlled [CPU trace](smb-level1-1-ppo-parity-trace-20261027-cpu.md) matched numerically over one 8,192-transition rollout and all 160 optimizer steps from cloned state. That bounded CPU result does not establish CUDA parity. The deferred CUDA diagnostic was never admitted and was canceled. This execution-profile study used FP32 throughout; a separately authorized [precision trial](smb-level1-1-ppo-precision-20261028-stopped.md) subsequently tested FP16 and BF16 and was also stopped early.

[Per-run data](smb-level1-1-ppo-profile-ablation-20261027-stopped-runs.csv) · [All verified checkpoints and public weight links](smb-level1-1-ppo-profile-ablation-20261027-stopped-checkpoints.csv). Existing MLflow frozen-policy quality companion runs retain the previously projected 96 points; the final post-stop collected point is retained in the portable data. Original results, videos, scientific receipts, failed observer logs and cancellation/resource-release records remain preserved.
