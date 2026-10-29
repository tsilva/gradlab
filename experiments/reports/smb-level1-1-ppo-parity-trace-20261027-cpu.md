# SMB PPO step-by-step parity diagnostic

The first full CPU rollout and PPO update matched exactly. No numerical divergence was observed. The user stopped the execution-profile study on 2026-10-28 and canceled the deferred CUDA comparison before admission. No CUDA diagnostic was run; the CPU evidence remains available below.

## What was compared

Both implementations started from one cloned NatureCNN/shared actor-critic policy and Adam state, using seed 100, the same Mario Level1-1 observations, the same per-action Python/NumPy/Torch RNG states, and equivalent independently advanced NumPy shuffle generators. The diagnostic used the real saved environment/reward contract and ROM identity, 16 lanes, 512 vector steps (8,192 environment transitions), batch size 512, and 10 epochs. Native provider and Torch CPU threads were limited to one for this diagnostic.

The production SB3 and GradLab GAE/update functions were called without editing them. The harness compared the two paths in alternating stages and stopped both at each comparison. It retained SB3's original Adam defaults and applied the custom backend's non-fused configuration only to GradLab. SB3 records `fused=None` and GradLab records `fused=False`; both produced identical CPU results.

| Stage | Comparisons | Result |
|---|---:|---|
| Initial weights and optimizer state | 1 each | Exact match |
| Rollout logits, actions, values and log-probabilities | 512 vector steps | Exact match; action RNG consumption equal |
| Rewards and rollout-end values | 512 steps + final values | Exact match |
| GAE recurrence and final advantages/returns | 512 reverse steps + final arrays | Exact match |
| Epoch shuffle order | 10 | Exact match |
| Minibatch inputs and forward outputs | 160 each | Exact match |
| KL guard, loss, raw/clipped gradients | 160 each | Exact match |
| Parameters and Adam moments before/after optimizer step | 160 each | Exact match |
| Final weights and Adam state | 1 each | Exact match |

“Exact match” here means elementwise numerical equality checked with `torch.equal`, with storage dtype differences recorded separately; it is not a separate signed-zero bit-pattern test. All 160 evaluated minibatches produced an optimizer step in both paths. There was no target-KL control-flow split. The full CPU pass took 163.77 seconds; this instrumented runtime is not a training throughput benchmark.

## Limits and canceled follow-up

This is one controlled CPU rollout/update from cloned state. It does not identify the first divergence in the earlier independent training Runs, verify their unrecorded initial hashes, or establish end-to-end CUDA reproducibility. No time-limit truncation occurred in this rollout, so the per-lane versus batched truncation-value branch was not exercised. The CPU result narrows the investigation but does not eliminate a GPU numerical, stochastic, initialization-path, or later-update difference.

The canceled follow-up had planned to compare the same production paths on the frozen CUDA runtime, first with production defaults and then with deterministic algorithms enabled and TF32 disabled as a control. Neither case was admitted. Any future CUDA comparison requires a new authorization. This CPU diagnostic confers no Acceptance, Promotion, or Release.

## Reproducibility

- Frozen GradLab source: `2069964e7ce38de8f87392de084145e6af8f87d3`.
- Frozen CUDA runtime: `docker:ghcr.io/tsilva/gradlab/gradlab-train@sha256:e5b0b2978a608984c6d499d272de11081c6b8b07a7c9354a9bdcc865071cc1a3`.
- CPU dependencies: SB3 `2.9.0`, Torch `2.13.0`, NumPy `2.4.6`, Mario provider `0.7.4`.
- Full CPU harness SHA256: `3033c82dd5a758b66f7eea3e1ecf6961600ad59cf891577f884f2f29b719649c`.
- CPU result SHA256: `6ab54a57d68fa338d3afbca83636fc02261cee8c202d8328e624b9739b9306fc`.
- The final CUDA harness adds explicit control-flow divergence handling and full NumPy-state checks; its short CPU smoke also matched. Its frozen SHA256 is `afdcdfb37958117df81212368bf030fba3d2693ff858017d0f85857c033f8147`.

[Complete CPU trace](smb-level1-1-ppo-parity-trace-20261027-cpu.json) · [Exact CPU harness](smb-level1-1-ppo-parity-trace-20261027-cpu.py) · [Diagnostic configuration](smb-level1-1-ppo-parity-trace-20261027-config.json)

Run the CPU harness from the frozen source/dependency environment with the configuration supplied through `--plan`, a fresh directory through `--output`, and `--device cpu --steps 512 --epochs 10`. The operator must supply their registered Mario ROM cache. No credentials are needed.
