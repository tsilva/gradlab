"""Offline categorical policy distillation from start-conditioned demonstrations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from gradlab.action_contract import (
    action_contract_entry,
    action_contract_meanings,
    assert_action_contract_compatible,
)
from gradlab.distillation_policy import Student, pixels
from gradlab.env import make_eval_vec_env, resolve_env_config
from gradlab.env_metadata import env_config_from_config_dict
from gradlab.env_registry import environment_spec
from gradlab.eval_metrics import (
    drain_episode_records,
    episode_is_complete,
    episode_result_from_record,
)
from gradlab.file_utils import atomic_write_json, file_sha256
from gradlab.policy_bundle import evaluation_contract, load_policy_bundle
from gradlab.policy_execution import verify_policy_execution_contract
from gradlab.policy_models import load_policy_model
from gradlab.policy_registry import resolve_policy_algorithm
from gradlab.policy_runtime import bind_policy_action_space
from gradlab.preprocessing import preprocessing_contract
from gradlab.rom_runtime import bind_rom_path
from gradlab.trusted_inputs import stage_model_input, verify_staged_model


@dataclass
class Teacher:
    level: str
    model: Any
    config: Any
    rom: Any
    saved_actions: Mapping[str, Any]
    watchdog: int
    identity: dict[str, Any]
    actions: tuple[str, ...] = ()

    @torch.inference_mode()
    def distribution(self, obs: Any) -> torch.Tensor:
        policy = self.model.policy
        tensor, _ = policy.obs_to_tensor(obs)
        dist = policy.get_distribution(tensor).distribution
        if not isinstance(dist, torch.distributions.Categorical):
            raise ValueError(f"{self.level} requires a categorical controller distribution")
        return dist.probs


def rollouts(
    teacher: Teacher,
    policy: Callable[[Any], torch.Tensor],
    *,
    lanes: int,
    seed: int,
    record: bool = False,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], int]:
    """Run one fresh episode per lane and optionally retain cleared episodes."""
    env = make_eval_vec_env(teacher.config, n_envs=lanes, seed=seed, rom_binding=teacher.rom)
    semantics = environment_spec(teacher.config.env_provider, teacher.config.game).eval_semantics
    frames, choices = [], []
    clears = np.zeros(lanes, dtype=bool)
    lengths = np.zeros(lanes, dtype=np.int32)
    waiting = np.ones(lanes, dtype=bool)
    torch.manual_seed(seed)
    try:
        runtime_contract = env.runtime.action_contract
        assert_action_contract_compatible(teacher.saved_actions, runtime_contract)
        actions = action_contract_meanings(runtime_contract)
        if not actions or env.action_space.n != len(actions) or env.action_space.start != 0:
            raise ValueError("expected a discrete zero-based controller vocabulary")
        if teacher.actions and teacher.actions != actions:
            raise ValueError("runtime controller vocabulary changed")
        teacher.actions = actions
        bind_policy_action_space(teacher.model, env.action_space, runtime_contract)
        verify_policy_execution_contract(teacher.model, env)
        env.seed(seed)
        observation = env.reset()
        for step in range(1, teacher.watchdog + 1):
            distribution = policy(observation)
            if distribution.shape != (lanes, len(actions)):
                raise ValueError("policy output does not match the controller vocabulary")
            if record:
                frames.append(pixels(observation).copy())
                choices.append(distribution.cpu().numpy().astype(np.float16))
            observation, _, _, infos = env.step(
                torch.multinomial(distribution, 1).squeeze(1).cpu().numpy()
            )
            for boundary in drain_episode_records(env):
                lane = int(boundary.lane)
                if waiting[lane]:
                    clears[lane] = episode_is_complete(
                        episode_result_from_record(
                            boundary,
                            semantics=semantics,
                            terminal_info=infos[lane],
                        )
                    )
                    lengths[lane] = step
                    waiting[lane] = False
            if not waiting.any():
                break
        if waiting.any():
            raise RuntimeError(
                f"{teacher.level}: {int(waiting.sum())} lanes expired without an episode boundary"
            )
    finally:
        env.close()
    retained = (
        [
            (
                np.stack([frame[lane] for frame in frames[: lengths[lane]]]),
                np.stack([choice[lane] for choice in choices[: lengths[lane]]]),
            )
            for lane in np.flatnonzero(clears)
        ]
        if record
        else []
    )
    return clears, retained, int(lengths.sum())


def local_actions(evidence: dict[str, Any], level: str) -> tuple[str, ...]:
    return tuple(evidence.get("level_actions", {}).get(level, evidence["actions"]))


def action_columns(teachers: list[Teacher], actions: tuple[str, ...]) -> list[tuple[int, ...]]:
    return [tuple(actions.index(action) for action in teacher.actions) for teacher in teachers]


def validate_teacher_compatibility(teachers: list[Teacher]) -> None:
    preprocessing = preprocessing_contract(teachers[0].config)
    controls: dict[str, Any] = {}
    for teacher in teachers:
        if preprocessing_contract(teacher.config) != preprocessing:
            raise ValueError(f"{teacher.level} has incompatible observation preprocessing")
        contract = teacher.saved_actions
        if contract["policy"].get("conditional_overrides"):
            raise ValueError(f"{teacher.level} uses unsupported conditional action overrides")
        for index, name in enumerate(teacher.actions):
            entry = action_contract_entry(contract, index)
            if entry is None or controls.setdefault(name, entry["controls"]) != entry["controls"]:
                raise ValueError(f"teacher semantics disagree for action {name!r}")


def balanced_batch(
    shards: list[tuple[np.ndarray, np.ndarray]],
    rows_per_level: int,
    rng: np.random.Generator,
    device: str,
    *,
    columns: list[tuple[int, ...]],
    action_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_rows, target_rows = [], []
    for (images, targets), action_ids in zip(shards, columns, strict=True):
        indices = rng.integers(len(images), size=rows_per_level)
        image_rows.append(images[indices])
        aligned = np.zeros((rows_per_level, action_count), dtype=np.float16)
        aligned[:, action_ids] = targets[indices]
        target_rows.append(aligned)
    images = torch.from_numpy(np.concatenate(image_rows)).to(device)
    soft_targets = torch.from_numpy(np.concatenate(target_rows)).to(
        device=device, dtype=torch.float32
    )
    levels = torch.arange(len(shards), device=device).repeat_interleave(rows_per_level)
    return images, levels, soft_targets / soft_targets.sum(dim=-1, keepdim=True)


def assess(
    teachers: list[Teacher],
    student: Student | None,
    *,
    episodes: int,
    lanes: int,
    seed: int,
    device: str,
    temperature: float = 1.0,
    columns: list[tuple[int, ...]] | None = None,
) -> dict[str, list[int]]:
    results = {}
    for index, teacher in enumerate(teachers):
        action_ids = torch.tensor(columns[index], device=device) if columns is not None else None

        @torch.inference_mode()
        def distribution(obs: Any) -> torch.Tensor:
            batch = torch.from_numpy(pixels(obs)).to(device)
            levels = torch.full((len(batch),), index, device=device, dtype=torch.long)
            logits = student(batch, levels)
            if action_ids is not None:
                logits = logits.index_select(dim=1, index=action_ids)
            return (logits / temperature).softmax(dim=-1)

        policy = teacher.distribution if student is None else distribution
        panel = []
        for start in range(0, episodes, lanes):
            clears, _, _ = rollouts(
                teacher, policy, lanes=min(lanes, episodes - start), seed=seed + start
            )
            panel.extend(int(value) for value in clears)
        results[teacher.level] = panel
    return results


def counts(panel: dict[str, list[int]]) -> dict[str, int]:
    return {level: sum(clears) for level, clears in panel.items()}


def write_recording_manifest(root: Path, evidence: dict[str, Any]) -> None:
    atomic_write_json(root / "dataset.json", evidence)


def write_recording_index(root: Path, levels: list[str]) -> None:
    index = {
        f"{level}/{name}": file_sha256(root / level / name)
        for level in levels
        for name in ("images.bin", "probabilities.bin", "episodes.json")
    }
    atomic_write_json(root / "shards.json", index)


def collect_recordings(
    output: Path,
    teachers: list[Teacher],
    *,
    episodes: int,
    lanes: int,
    seed: int,
    event: Callable[[str], None],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    shards = []
    level_actions = {teacher.level: teacher.actions for teacher in teachers}
    if any(not ids or len(ids) != len(set(ids)) for ids in level_actions.values()):
        raise ValueError("teachers require ordered, unique controller semantics")
    actions = tuple(dict.fromkeys(action for ids in level_actions.values() for action in ids))
    evidence: dict[str, Any] = {
        "levels": [teacher.level for teacher in teachers],
        "teachers": {teacher.level: teacher.identity for teacher in teachers},
        "collection": {},
        "actions": actions,
        "level_actions": level_actions,
        "collection_seed": seed,
        "lanes": lanes,
    }
    for teacher in teachers:
        shard_root = output / teacher.level
        shard_root.mkdir()
        spans = []
        retained_rows = 0
        image_shape = None
        cleared = transitions = 0
        with (
            (shard_root / "images.bin").open("xb") as image_file,
            (shard_root / "probabilities.bin").open("xb") as target_file,
        ):
            for start in range(0, episodes, lanes):
                result, retained, steps = rollouts(
                    teacher,
                    teacher.distribution,
                    record=True,
                    lanes=min(lanes, episodes - start),
                    seed=seed + start,
                )
                cleared += int(result.sum())
                transitions += steps
                required = sum(
                    images.nbytes + probabilities.nbytes for images, probabilities in retained
                )
                if shutil.disk_usage(output).free < required + 2 * 1024**3:
                    raise RuntimeError("insufficient recording disk headroom")
                for lane, (frames, probabilities) in zip(
                    np.flatnonzero(result), retained, strict=True
                ):
                    if image_shape is None:
                        image_shape = frames.shape[1:]
                    if frames.shape[1:] != image_shape:
                        raise ValueError("pixel observation layout changed within a level")
                    spans.append(
                        {
                            "batch_seed": seed + start,
                            "lane": int(lane),
                            "offset": retained_rows,
                            "length": len(frames),
                        }
                    )
                    frames.tofile(image_file)
                    probabilities.tofile(target_file)
                    retained_rows += len(frames)
        if image_shape is None:
            raise RuntimeError(f"{teacher.level} produced no clearing episodes")
        if shards and image_shape != shards[0][0].shape[1:]:
            raise ValueError("teachers do not share a pixel observation layout")
        shards.append(
            (
                np.memmap(
                    shard_root / "images.bin",
                    dtype=np.uint8,
                    mode="r",
                    shape=(retained_rows, *image_shape),
                ),
                np.memmap(
                    shard_root / "probabilities.bin",
                    dtype=np.float16,
                    mode="r",
                    shape=(retained_rows, len(teacher.actions)),
                ),
            )
        )
        (shard_root / "episodes.json").write_text(json.dumps(spans, indent=2) + "\n")
        evidence["collection"][teacher.level] = {
            "episodes": episodes,
            "clears": cleared,
            "transitions": transitions,
            "retained_transitions": retained_rows,
        }
        evidence["image_shape"] = shards[0][0].shape[1:]
        write_recording_manifest(output, evidence)
        event(f"distillation collection {teacher.level}: {cleared}/{episodes} completed")

    write_recording_index(output, evidence["levels"])
    return shards, evidence


def recording_shard(
    root: Path, evidence: dict[str, Any], level: str
) -> tuple[np.ndarray, np.ndarray]:
    rows = evidence["collection"][level]["retained_transitions"]
    arrays = []
    for name, dtype, shape in (
        ("images.bin", np.uint8, (rows, *evidence["image_shape"])),
        ("probabilities.bin", np.float16, (rows, len(local_actions(evidence, level)))),
    ):
        path = root / level / name
        if rows < 1 or path.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
            raise ValueError(f"incomplete recording shard at {level}/{name}")
        arrays.append(np.memmap(path, dtype=dtype, mode="r", shape=shape))
    return arrays[0], arrays[1]


def load_teacher(
    level: str,
    identity: dict[str, str],
    bundle_root: Path,
    *,
    device: str,
    game: str,
    rom_binding: Any | None,
) -> Teacher:
    """Load a staged, recipe-pinned teacher through GradLab's usual trust boundary."""
    source = f"{identity['ref']}@{identity['revision']}"
    staged = stage_model_input(bundle_root / "model.zip", source_identity=source)
    try:
        with verify_staged_model(staged) as approved:
            bundle = load_policy_bundle(
                staged.root,
                source=identity["ref"],
                revision=identity["revision"],
            )
            if (
                bundle.checkpoint_sha256 != identity["checkpoint_sha256"]
                or bundle.recipe_sha256 != identity["recipe_sha256"]
            ):
                raise ValueError(f"{level} teacher does not match the recipe's digests")
            recorded = evaluation_contract(bundle.recipe)
            config = resolve_env_config(env_config_from_config_dict(recorded["environment"]))
            if config.game != game or config.state != level:
                raise ValueError(f"{level} teacher does not describe the declared game and start")
            if recorded["action_sampling"] != "stochastic":
                raise ValueError(f"{level} collection requires a stochastic categorical teacher")
            asset = recorded.get("asset")
            if asset:
                if rom_binding is None:
                    raise ValueError(f"{level} requires the declared ROM")
                rom = bind_rom_path(asset, rom_binding.path)
            else:
                rom = None
            algorithm = resolve_policy_algorithm(bundle.model["policy"])
            if algorithm not in {"ppo", "a2c"}:
                raise ValueError(
                    f"{level} teacher does not expose the supported actor distribution"
                )
            action_contract = bundle.model["provenance"]["training_metadata"]["action_contract"]
            actions = action_contract_meanings(action_contract)
            if not actions or len(set(actions)) != len(actions):
                raise ValueError(f"{level} teacher has no unique categorical action semantics")
            model = load_policy_model(approved, device=device, algorithm_id=algorithm)
    finally:
        staged.cleanup()
    model.policy.set_training_mode(False)
    return Teacher(
        level,
        model,
        config,
        rom,
        action_contract,
        int(recorded["watchdog_steps"]),
        identity,
        actions,
    )


def verified_recordings(
    root: Path,
    teachers: list[Teacher],
    *,
    manifest_sha256: str,
    index_sha256: str,
    event: Callable[[str], None],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    manifest = root / "dataset.json"
    index = root / "shards.json"
    if file_sha256(manifest) != manifest_sha256 or file_sha256(index) != index_sha256:
        raise ValueError(
            "distillation dataset manifest or shard index digest disagrees with recipe"
        )
    evidence = json.loads(manifest.read_bytes())
    if evidence["teachers"] != {teacher.level: teacher.identity for teacher in teachers}:
        raise ValueError("distillation dataset binds different teachers")
    levels = [teacher.level for teacher in teachers]
    if evidence["levels"] != levels or set(evidence["collection"]) != set(levels):
        raise ValueError("distillation dataset level order or coverage disagrees with recipe")
    if any(local_actions(evidence, teacher.level) != teacher.actions for teacher in teachers):
        raise ValueError("distillation dataset has incompatible teacher-action semantics")
    expected = {
        f"{level}/{name}"
        for level in levels
        for name in ("images.bin", "probabilities.bin", "episodes.json")
    }
    file_index = json.loads(index.read_bytes())
    if set(file_index) != expected:
        raise ValueError("distillation shard index has an incomplete or unexpected file set")
    for level in levels:
        for name in ("images.bin", "probabilities.bin", "episodes.json"):
            relative = f"{level}/{name}"
            path = root / relative
            if not path.is_file() or path.is_symlink() or path.parent.is_symlink():
                raise ValueError(f"distillation shard is not a regular file: {relative}")
            if file_sha256(path) != file_index[relative]:
                raise ValueError(f"distillation shard has changed: {relative}")
        event(f"verified distillation recordings: {level}")
    return [recording_shard(root, evidence, level) for level in levels], evidence
