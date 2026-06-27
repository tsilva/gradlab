# 2026-06-27 Peak Checkpoint Eval Ranking

Goal metric: maximize `train/info/level_complete/rate/min/last` for the
Level1-1/Level1-2 mixed policy. This memo stores the checkpoint references and
100-episode eval results for the top training-ranked candidates.

## Eval Protocol

Selection source: full-history W&B scan over the top mixed Level1-1/Level1-2
batches, ranked by per-run peak
`train/info/level_complete/rate/min/last`. These historical runs also had the
legacy key `train/info/level_complete/rate_min/last`, so the scan treated that
as a fallback alias.

For each ranked batch, the evaluated checkpoint was the nearest uploaded
checkpoint to the selected run's training peak.

Eval command template:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-eval \
  --artifact <artifact> \
  --episodes 100 \
  --n-envs 64 \
  --device cpu \
  --force
```

Eval contract for this comparison:

- 100 episodes.
- CPU inference with `--n-envs 64`.
- Task conditioning loaded from the artifact/run metadata.
- Evaluation continues after level changes; level completion is recorded from
  info instead of terminating the rollout on completion.
- Life-loss termination disabled for eval.

Primary eval ranking metric:
`eval/info/level_complete/rate/min/last`, with
`eval/info/level_complete/rate/mean/last` as the companion/tiebreak metric.

## Training Peak References

| Training rank | Batch | Spec | Batch peak mean | Batch peak std | Selected run | Run id | Train peak min | Peak step | Eval checkpoint |
| ---: | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| 1 | `b93-l11l12-slowent-l12bias-two-seed` | `b93-slowent-l12bias-l11l12-two-seed` | `0.905` | `0.0495` | `b93_l11l12_slowent_l12bias_s201_20260627T112704Z` | `vvmq6arm` | `0.94` | `4,967,360` | `tsilva/SuperMarioBros-NES/b93_l11l12_slowent_l12bias_s201_20260627T112704Z-checkpoint:step-5000000` |
| 2 | `b114-l11l12-lowpress-complete25-slowent-l12micro-two-seed` | `b114-lowpress-complete25-slowent-l12micro-l11l12-two-seed` | `0.89` | `0.0283` | `b114_l11l12_lowpress_complete25_slowent_l12micro_s206_20260627T141519Z` | `zpjlp3nt` | `0.91` | `4,862,176` | `tsilva/SuperMarioBros-NES/b114_l11l12_lowpress_complete25_slowent_l12micro_s206_20260627T141519Z-checkpoint:step-5000000` |
| 3 | `b97-l11l12-complete25-slowent-two-seed` | `b97-complete25-slowent-l11l12-two-seed` | `0.865` | `0.0071` | `b97_l11l12_complete25_slowent_s198_20260627T114643Z` | `148sau66` | `0.87` | `4,106,976` | `tsilva/SuperMarioBros-NES/b97_l11l12_complete25_slowent_s198_20260627T114643Z-checkpoint:step-4000000` |
| 4 | `b115-l11l12-slowent-l12bias-complete25-two-seed` | `b115-slowent-l12bias-complete25-l11l12-two-seed` | `0.805` | `0.0354` | `b115_l11l12_slowent_l12bias_complete25_s201_20260627T142436Z` | `gz5rfj51` | `0.83` | `4,236,944` | `tsilva/SuperMarioBros-NES/b115_l11l12_slowent_l12bias_complete25_s201_20260627T142436Z-checkpoint:step-4000000` |
| 5 | `b91-l11l12-lowpress-complete25-two-seed` | `b91-lowpress-complete25-l11l12-two-seed` | `0.775` | `0.1485` | `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | `al1h7yu0` | `0.88` | `4,089,456` | `tsilva/SuperMarioBros-NES/b91_l11l12_lowpress_complete25_s206_20260627T112013Z-checkpoint:step-4000000` |

## Eval Ranking

| Eval rank | Batch | Run id | Eval min | Eval mean | Level-change done rate | Level1-1 rate | Level1-2 rate | Reward mean | Best x |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | B114 | `zpjlp3nt` | `0.9661016949152542` | `0.9708557255064076` | `0.97` | `0.975609756097561` (`n=41`) | `0.9661016949152542` (`n=59`) | `3379.277501044944` | `9383` |
| 2 | B93 | `vvmq6arm` | `0.9310344827586207` | `0.94439048081593` | `0.95` | `0.9310344827586207` (`n=29`) | `0.9577464788732394` (`n=71`) | `3417.492001552172` | `9387` |
| 3 | B91 | `al1h7yu0` | `0.9047619047619048` | `0.9265188834154352` | `0.93` | `0.9047619047619048` (`n=42`) | `0.9482758620689655` (`n=58`) | `3268.4845011039824` | `9380` |
| 4 | B97 | `148sau66` | `0.8793103448275862` | `0.9039408866995073` | `0.90` | `0.9285714285714286` (`n=42`) | `0.8793103448275862` (`n=58`) | `3322.826501330398` | `6761` |
| 5 | B115 | `gz5rfj51` | `0.8125` | `0.8841911764705883` | `0.91` | `0.8125` (`n=32`) | `0.9558823529411765` (`n=68`) | `3280.86850117404` | `9392` |

## W&B And Local Summary References

| Batch | W&B run | Local eval summary |
| --- | --- | --- |
| B114 | `https://wandb.ai/tsilva/SuperMarioBros-NES/runs/zpjlp3nt` | `wandb/run-20260627_225829-zpjlp3nt/files/wandb-summary.json` |
| B93 | `https://wandb.ai/tsilva/SuperMarioBros-NES/runs/vvmq6arm` | `wandb/run-20260627_225337-vvmq6arm/files/wandb-summary.json` |
| B91 | `https://wandb.ai/tsilva/SuperMarioBros-NES/runs/al1h7yu0` | `wandb/run-20260627_231302-al1h7yu0/files/wandb-summary.json` |
| B97 | `https://wandb.ai/tsilva/SuperMarioBros-NES/runs/148sau66` | `wandb/run-20260627_230441-148sau66/files/wandb-summary.json` |
| B115 | `https://wandb.ai/tsilva/SuperMarioBros-NES/runs/gz5rfj51` | `wandb/run-20260627_230907-gz5rfj51/files/wandb-summary.json` |

## Interpretation

B114 is the eval winner even though B93 had the best training high-watermark.
B93 remained strong, but B114 had better balanced eval min and mean.

B91's selected peak checkpoint evaluated surprisingly well and ranked above B97
by eval min, despite B91 being a less reliable two-seed batch: seed `206`
peaked high, while seed `207` peaked at `0.67` and crashed.

B97 was consistent during training, but its peak-nearest checkpoint eval landed
behind B114, B93, and B91.

B115 was Level1-1-bottlenecked in eval. It had a strong Level1-2 rate
(`0.9558823529411765`) but only `0.8125` from Level1-1 starts.

B116, the five-seed B93 repro batch, did not reproduce B93 robustly across more
seeds: peak values were `0.93`, `0.88`, `0.39`, `0.31`, and `0.09` with batch
mean `0.52` and std `0.369`.
