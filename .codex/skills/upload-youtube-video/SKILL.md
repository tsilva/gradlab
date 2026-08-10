---
name: upload-youtube-video
description: Upload or update YouTube model preview videos from gradlab, especially RL gameplay previews associated with Hugging Face model cards. Use when the user asks to upload a local video to YouTube, update an existing video description/title/tags/privacy, add a video to a playlist, produce OAuth authorization flow instructions, or ensure model-card preview videos have clean clickable model links.
---

# Upload YouTube Video

## Contract

Publish a local preview video to the user's YouTube account, connect it to the associated model page, and leave enough metadata for future agents to update or verify it.

For player-initiated checkpoint publication, treat the admitted request as immutable. Upload the
exact admitted `replay.mp4`, add the hidden `gradlab-publication-<request-fingerprint>` tag, and
verify successful YouTube processing before Hugging Face publication starts. Never infer
evaluation evidence from the replay itself.

Do not expose or admit player publication while the current episode is unfinished or its replay
is still encoding. A prior completed capture may remain visible as provenance, but must become
ineligible as soon as a new episode begins and must not substitute for that current episode.

Prefer the repo uploader script at `scripts/upload_youtube_video.py`. It reads `youtube_client_secret.json` and stores `youtube_token.json` in GradLab's user configuration directory (`~/.config/gradlab` by default, or `$XDG_CONFIG_HOME/gradlab` when configured); do not print secret or token contents. Explicit `--client-secret` and `--token` paths remain available for operator-directed overrides.

## Before Uploading

1. Confirm the local video path, generated title, immutable model tag URL, and requested final privacy status. New research releases upload as unlisted until both destinations and containers verify.
2. Use the generated environment playlist `GradLab — <canonical environment ID>`; only editorial cross-environment curation uses `GradLab — Featured Research`. Do not expose an editable playlist field.
3. Check the model page URL is direct and fully qualified. Prefer `https://huggingface.co/<owner>/<repo>` over redirect/short domains such as `https://hf.co/...`.
4. For gameplay preview videos, set a custom YouTube thumbnail from the local video at 10 seconds by default. The repo uploader does this automatically unless `--no-thumbnail` is passed. If the video is shorter than 10 seconds, use the last safe frame the uploader selects.
5. If the user needs OAuth again, run the uploader with `--no-browser` only when appropriate and relay the printed authorization URL. After the user says they authorized it, continue the waiting process; do not restart unless the callback failed.

## Description Format

Use the generated research-release title shape:

```text
<materialized goal title> — <Trainer> <ALGORITHM> @ <compact env steps>
```

If it exceeds YouTube's limit, use `<canonical-env-id> / <goal-id> — <Trainer> <ALGORITHM> @
<compact env steps>` and fail rather than truncating if that fallback is still too long. Keep
acceptance and replay outcomes out of the title. Identify the actual trainer, never its
compatibility layer.

Use this description shape for RL/model preview videos:

```text
A <ALGORITHM> reinforcement-learning agent trained with <Trainer> <plays/completes> <Game Display Name>[ <Level/State Display Name>] with a <win-rate> verified evaluation win rate.

Model: https://huggingface.co/<owner>/<repo>
gradlab: https://github.com/tsilva/gradlab

#ReinforcementLearning #<ALGORITHM> #<GameName>
```

Rules:

- Include the exact checkpoint step and identity, evaluation protocol/count, every acceptance
  outcome, key ranking metrics, separately labeled replay result, immutable repo-wide HF `vN`
  tag, W&B, R2 manifest, source
  commit/recipe, GitHub, environment collection, and environment playlist in the description.
- Never invent generic success or win-rate claims; render the materialized goal metrics documented
  by `METRICS.md`.
- Prefer human display names over raw env ids in titles/descriptions. For example, use `Super Mario Bros NES` instead of `SuperMarioBros-Nes-v0`, and `Level 1-2` instead of `Level1-2`.
- Start with a concise human-readable description of what the video shows.
- Mention that the checkpoint was trained with `gradlab` in the first sentence when true.
- Put the `Model:` link before the `gradlab:` link.
- Keep the link labels short and plain: `gradlab:` and `Model:`.
- Include up to three visible hashtags at the end of the description: `#ReinforcementLearning`, `#<ALGORITHM>`, and one compact game hashtag such as `#SuperMarioBros` or `#MegaMan`.
- Add hidden tags for search/context: reinforcement learning, deep reinforcement learning, AI
  gameplay, actual training backend, gradlab, algorithm, raw game id, human game name, raw
  level/state id, and human level/state name. Add `stable-baselines3` only for an SB3 training
  backend and `Stable Retro` only for a Stable Retro environment provider.
- Avoid shorteners and redirect domains for YouTube descriptions; they can be visually truncated or treated suspiciously by YouTube.
- Expect YouTube to visually ellipsize long links in collapsed views even when the actual link is correct and clickable.
- Include extra eval claims only when the user asks for them or when they are needed for the video description and are backed by the current model card, private-R2 evaluation evidence, W&B metrics, or generated summary files.
- Keep evidence-bearing title text, replay outcome, evaluation statements, the direct model URL,
  and the gradlab link generated and locked in player publication. An optional operator note must
  be labeled `Operator note (not verified evidence)` and kept separate from generated claims.
- Player publication has no destination opt-outs. YouTube must reach `processingStatus=succeeded`
  while unlisted with the admitted channel, marker, and metadata before the Hugging Face mutation
  begins. Set requested privacy only after HF, both environment containers, metadata, and thumbnail
  verify.
- Persist a resumable session before uploading bytes and reconcile it before retrying. If the final
  response is uncertain, do not automatically upload a duplicate; require an explicit verified
  attachment or an acknowledged abandon-and-reupload decision.

## Upload Command

Use the script from the repo root:

```bash
python3 scripts/upload_youtube_video.py <video.mp4> \
  --title "<Game Display Name> — <Distinct Level/State Display Name> — <Trainer> PPO" \
  --human-description "A PPO reinforcement-learning agent trained with <Trainer> completes <Game Display Name> <Level/State Display Name> with a <win-rate> verified evaluation win rate." \
  --model-page $'Model: https://huggingface.co/<owner>/<repo>\ngradlab: https://github.com/tsilva/gradlab' \
  --description "#ReinforcementLearning #PPO #<GameName>" \
  --tags "reinforcement learning,deep reinforcement learning,AI gameplay,stable-baselines3,Stable Retro,gradlab,PPO,<raw game id>,<Game Display Name>,<raw level/state id>,<Level/State Display Name>" \
  --playlist-title gradlab \
  --privacy-status public \
  --output ~/.config/gradlab/runs/<artifact-dir>/youtube_upload_result.json
```

The script captures and uploads a custom thumbnail from second `10` of the video by default. Override with `--thumbnail-time <seconds>`, provide an explicit `--thumbnail <image.jpg>`, or disable with `--no-thumbnail`.

If the OAuth callback cannot open a browser or the user asks for the OAuth URL:

```bash
python3 scripts/upload_youtube_video.py <video.mp4> ... --no-browser
```

Relay the URL printed by the script and wait for the user to authorize.

## Updating an Existing Video

Use `--video-id` to update metadata without re-uploading:

```bash
python3 scripts/upload_youtube_video.py \
  --video-id <youtube-video-id> \
  --title "<Game Display Name> — <Distinct Level/State Display Name> — <Trainer> PPO" \
  --human-description "A PPO reinforcement-learning agent trained with <Trainer> completes <Game Display Name> <Level/State Display Name> with a <win-rate> verified evaluation win rate." \
  --model-page $'Model: https://huggingface.co/<owner>/<repo>\ngradlab: https://github.com/tsilva/gradlab' \
  --description "#ReinforcementLearning #PPO #<GameName>" \
  --privacy-status public \
  --output ~/.config/gradlab/runs/<artifact-dir>/youtube_upload_result.json \
  --no-browser
```

The update path preserves existing title and tags unless replacements are passed.

## Verification

After upload or update:

1. Read the JSON output and report the YouTube URL, playlist URL when present, and the final description shape.
2. Verify the model link is direct `https://huggingface.co/...` and appears on a `Model:` line.
3. For new gameplay uploads, verify the JSON includes `thumbnail.path` and a successful `thumbnail.upload` response, or report the thumbnail API failure clearly if the account does not allow custom thumbnails.
4. Run `python3 -m py_compile scripts/upload_youtube_video.py` after changing the uploader script.
5. If the user reports link issues and the stored description is correct, distinguish visual truncation from actual clickability. For clickability issues, check account verification, Shorts vs normal video, direct URL formatting, redirect domains, and whether the issue is only in a collapsed/mobile view.

## Repo Rules

When YouTube upload policy or description conventions change, encode detailed rules here first. Keep `AGENTS.md` as a short trigger pointer to this skill, not the detailed source of truth.
