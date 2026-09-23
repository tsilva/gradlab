# Hugging Face research catalog

This page maps GradLab-related public work as of 2026-09-23. A **Policy** can be
played; a **Checkpoint** is a saved Policy state from a Run; a **Release** is an
immutable package of a promoted Policy and its Acceptance evidence. A dataset
contains recorded episodes for analysis or training. Dataset Publication does
not grant Acceptance or Promotion.

## Play a Policy

| Environment | Public entry | What it contains |
| --- | --- | --- |
| `VizdoomDeathmatch-v1` | [GradLab ViZDoom Policy](https://huggingface.co/tsilva/VizdoomDeathmatch-v1_gradlab-ppo_b0330247) | Current GradLab PPO Research Goal, immutable `v3` Release, Acceptance evidence, and Playback command. |
| `SuperMarioBros-Nes-v0` | [Mario environment collection](https://huggingface.co/collections/tsilva/gradlab-supermariobros-nes-v0-6a5675af108d798040f3aafb) | Nine historical `rlab` / Stable-Baselines3 PPO Policy Releases and a separate gameplay recording. These are not GradLab-trained Checkpoints. |

The [ViZDoom environment collection](https://huggingface.co/collections/tsilva/gradlab-vizdoomdeathmatch-v1-6a75be1f7f77460f66953c43) lists its current Release.

The historical Mario `rlab` Policy Releases are [Level1-1](https://huggingface.co/tsilva/Level1-1_stable-baselines3-ppo_7d9b9131), [Level1-2](https://huggingface.co/tsilva/Level1-2_stable-baselines3-ppo_91df110f), [Level1-4](https://huggingface.co/tsilva/Level1-4_stable-baselines3-ppo_be9abbc8), [Level2-1](https://huggingface.co/tsilva/Level2-1_stable-baselines3-ppo_4792f358), [Level2-3](https://huggingface.co/tsilva/Level2-3_stable-baselines3-ppo_ff330a59), [Level3-2](https://huggingface.co/tsilva/Level3-2_stable-baselines3-ppo_945d412a), [Level3-3](https://huggingface.co/tsilva/Level3-3_stable-baselines3-ppo_d52f7c68), [Level3-4](https://huggingface.co/tsilva/Level3-4_stable-baselines3-ppo_4141caab), and [Level4-1](https://huggingface.co/tsilva/Level4-1_stable-baselines3-ppo_fe8a02dc).

## Use recorded episodes

The [Breakout environment collection](https://huggingface.co/collections/tsilva/gradlab-breakout-atari2600-v0) groups its four datasets and two derived world models.

| Dataset | Source and image contract | Best starting point |
| --- | --- | --- |
| [Breakout collector trajectories](https://huggingface.co/datasets/tsilva/gradlab-breakout-trajectories) | Separate Policy collector; top 17 image rows blackened before storage. | Action-conditioned trajectories and reusable frame assets. |
| [Breakout raw Checkpoint evaluations](https://huggingface.co/datasets/tsilva/gradlab-breakout-checkpoint-evals) | Checkpoint Monitoring of Run `gradlab-8972634776926e147a00bf284420341c`; full HUD; episode index and linked chunks. | Inspect the original evaluation inventory and provenance. |
| [Breakout Checkpoint trajectories, Run `gradlab-6127e81d…`](https://huggingface.co/datasets/tsilva/gradlab-breakout-6127e81d) | Full-HUD Checkpoint Monitoring; 500 episodes; explicit derived split view. | Load `transitions`, `frames`, `episodes`, and `sessions` tables. |
| [Breakout Checkpoint trajectories, Run `gradlab-c6d579da…`](https://huggingface.co/datasets/tsilva/gradlab-breakout-c6d579da) | Full-HUD Checkpoint Monitoring; 1,000 episodes; unsplit and derived split views. | Load the `all` view or the documented train/validation/test view. |
| [Mario gameplay recording](https://huggingface.co/datasets/tsilva/NES-SuperMarioBros_Level1-1_gray84-hudcrop-stack4-simple_ppo) | One historical `gymrec` recording with its own collector and capture contract. | Inspect the episode and its preview. |

The two full-HUD Breakout trajectory datasets come from different Runs. The
HUD-masked collector dataset has a different capture contract. Do not combine
their images or results as one scientific dataset. Checkpoint Monitoring is
observational; its scores and recordings do not establish Acceptance.

The [Breakout world model](https://huggingface.co/tsilva/gymemu-breakout-unified-dynamics-fs1) and [state decoder](https://huggingface.co/tsilva/gymemu-breakout-state-decoder-fs1) are derived models trained from a GradLab dataset. They are not playable Policies or GradLab Releases.
