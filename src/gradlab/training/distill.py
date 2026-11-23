"""GradLab offline distillation training backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from gradlab.action_contract import runtime_action_contract
from gradlab.artifacts import install_model_bundle
from gradlab.distillation import (
    action_columns,
    assess,
    balanced_batch,
    collect_recordings,
    counts,
    load_teacher,
    validate_teacher_compatibility,
    verified_recordings,
)
from gradlab.distillation_policy import STUDENT_MODEL_CLASS, DistilledPolicy
from gradlab.env import make_eval_vec_env
from gradlab.file_utils import atomic_write_json
from gradlab.metric_names import (
    TRAIN_DISTILL_CROSS_ENTROPY,
    TRAIN_DISTILL_DEVELOPMENT_CLEAR,
    TRAIN_DISTILL_DEVELOPMENT_WORST_GAP,
)
from gradlab.training_backend import CHECKPOINT_EVAL_ACCEPTANCE, BackendContext
from gradlab.training_lifecycle import ProgressField, TerminalReason, TrainingResult

BACKEND_ID = "gradlab.distill"
DATASET_CACHE_ENV = "GRADLAB_DISTILLATION_INPUT"
CONTAINER_INPUT = "/distillation-input"

_SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_Count = Annotated[int, Field(gt=0)]
_Seed = Annotated[int, Field(ge=0, lt=2**32)]
_Temperature = Annotated[float, Field(gt=0, le=10, allow_inf_nan=False)]


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TeacherConfig(_Config):
    id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_][a-zA-Z0-9_-]{0,95}$")]
    repo_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")]
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    checkpoint_sha256: _SHA
    recipe_sha256: _SHA

    def identity(self) -> dict[str, str]:
        return {"ref": f"hf://{self.repo_id}", **self.model_dump(exclude={"id", "repo_id"})}


class DatasetConfig(_Config):
    manifest_sha256: _SHA
    shards_sha256: _SHA


class CollectionConfig(_Config):
    attempts_per_task: _Count
    seed_start: _Seed
    retain: Literal["cleared"] = "cleared"


class PhaseConfig(_Config):
    seed: _Seed
    learning_rate: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    evaluate_at_updates: Annotated[list[_Count], Field(min_length=1)]
    heldout_seed: _Seed | None = None


class EvaluationConfig(_Config):
    lanes: _Count
    development_episodes: _Count
    development_seed: _Seed
    development_temperature: _Temperature
    heldout_episodes: _Count
    final_seed: _Seed
    temperatures: Annotated[list[_Temperature], Field(min_length=1)]


class DistillationConfig(_Config):
    teachers: Annotated[list[TeacherConfig], Field(min_length=1, max_length=128)]
    dataset: DatasetConfig | None = None
    collection: CollectionConfig | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    architecture: Literal["cnn_3conv_mlp512"] = "cnn_3conv_mlp512"
    samples_per_task: _Count
    phases: Annotated[list[PhaseConfig], Field(min_length=1)]
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def validate_protocol(self) -> "DistillationConfig":
        if (self.dataset is None) == (self.collection is None):
            raise ValueError("choose exactly one frozen dataset or fresh collection")
        if len({teacher.id for teacher in self.teachers}) != len(self.teachers):
            raise ValueError("teacher start IDs must be unique")
        if any(
            phase.evaluate_at_updates != sorted(set(phase.evaluate_at_updates))
            for phase in self.phases
        ):
            raise ValueError("phase checkpoint updates must be strictly increasing")
        if len(set(self.evaluation.temperatures)) != len(self.evaluation.temperatures):
            raise ValueError("student temperatures must be unique")
        evaluation = self.evaluation
        intervals = [
            (evaluation.development_seed, evaluation.development_episodes),
            (evaluation.final_seed, evaluation.heldout_episodes),
            *(
                (phase.heldout_seed, evaluation.heldout_episodes)
                for phase in self.phases
                if phase.heldout_seed is not None
            ),
        ]
        if self.collection:
            intervals.append((self.collection.seed_start, self.collection.attempts_per_task))
        intervals.sort()
        if any(
            start + size > following[0]
            for (start, size), following in zip(intervals, intervals[1:])
        ):
            raise ValueError("collection, development, and held-out seed windows must be disjoint")
        if any(start + size > 2**32 for start, size in intervals):
            raise ValueError("distillation seed window exceeds its range")
        return self

    @property
    def total_updates(self) -> int:
        return sum(phase.evaluate_at_updates[-1] for phase in self.phases)


def _score(teacher: dict[str, int], student: dict[str, int]) -> tuple[int, int]:
    return -max(teacher[level] - student[level] for level in teacher), sum(student.values())


def _weights(student: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}


def _fit_phase(
    context: BackendContext,
    config: DistillationConfig,
    phase: PhaseConfig,
    student: torch.nn.Module,
    shards: list,
    columns: list[tuple[int, ...]],
    action_count: int,
    assess_student: Any,
    dev_teacher: dict[str, int],
    *,
    continuation: bool,
    step: int,
    device: str,
) -> tuple[int, list[dict], int]:
    rng = np.random.default_rng(phase.seed)
    optimizer = torch.optim.AdamW(student.parameters(), lr=phase.learning_rate)
    mask = torch.tensor(
        [[action in ids for action in range(action_count)] for ids in columns],
        device=device,
    )
    scores = []
    best_weights = _weights(student)
    best_score = None
    best_at = 0
    stale = 0
    mean_loss = 0.0
    interval_updates = 0
    checkpoints = ([0] if continuation else []) + phase.evaluate_at_updates
    position = 0
    for checkpoint in checkpoints:
        while position < checkpoint and not context.stop_flag.requested:
            images, tasks, labels = balanced_batch(
                shards,
                config.samples_per_task,
                rng,
                device,
                columns=columns,
                action_count=action_count,
            )
            logits = student(images, tasks)
            logits = logits.masked_fill(~mask[tasks], torch.finfo(logits.dtype).min)
            loss = -(labels * logits.log_softmax(dim=1)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            mean_loss += float(loss.detach())
            interval_updates += 1
            position += 1
            if position % 250 == 0 and position != checkpoint:
                context.session.report(
                    step=step + position,
                    metrics={TRAIN_DISTILL_CROSS_ENTROPY: mean_loss / interval_updates},
                )
                mean_loss = 0.0
                interval_updates = 0
        if context.stop_flag.requested:
            break
        clears = counts(
            assess_student(
                config.evaluation.development_seed,
                config.evaluation.development_episodes,
                config.evaluation.development_temperature,
            )
        )
        score = _score(dev_teacher, clears)
        scores.append({"updates": position, "clears": clears, "worst_deficit": -score[0]})
        metrics = {
            TRAIN_DISTILL_DEVELOPMENT_CLEAR: sum(clears.values())
            / (len(clears) * config.evaluation.development_episodes),
            TRAIN_DISTILL_DEVELOPMENT_WORST_GAP: -score[0] / config.evaluation.development_episodes,
        }
        if interval_updates:
            metrics[TRAIN_DISTILL_CROSS_ENTROPY] = mean_loss / interval_updates
        mean_loss = 0.0
        interval_updates = 0
        context.session.report(step=step + position, metrics=metrics)
        if best_score is None or score > best_score:
            best_weights, best_score, best_at, stale = _weights(student), score, position, 0
        else:
            stale += 1
        if (
            min(clears.values()) > 0 and -score[0] / config.evaluation.development_episodes <= 0.05
        ) or stale >= 2:
            break
        if stale == 1:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
    student.load_state_dict(best_weights)
    return position, scores, best_at


def run_distillation(context: BackendContext, configuration: DistillationConfig) -> TrainingResult:
    input_value = os.environ.get(DATASET_CACHE_ENV)
    if not input_value:
        raise ValueError(f"distillation requires staged teacher inputs via {DATASET_CACHE_ENV}")
    input_root = Path(input_value)
    if not input_root.is_absolute() or not input_root.is_dir():
        raise ValueError("distillation input root must be an absolute directory")
    if os.environ.get("GRADLAB_ORCHESTRATOR") == "dstack" and (
        input_root != Path(CONTAINER_INPUT) or not os.statvfs(input_root).f_flag & os.ST_RDONLY
    ):
        raise ValueError("dstack distillation requires the kernel read-only input mount")
    selected_device = configuration.device
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if selected_device == "auto"
        else selected_device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA distillation device is unavailable")
    torch.set_num_threads(2)
    context.session.event(f"using torch device: {device}")
    teachers = [
        load_teacher(
            teacher.id,
            teacher.identity(),
            input_root / "teachers" / teacher.id,
            device=device,
            game=context.environment.game,
            rom_binding=context.rom_binding,
        )
        for teacher in configuration.teachers
    ]
    validate_teacher_compatibility(teachers)
    evaluation = configuration.evaluation
    if configuration.dataset:
        shards, evidence = verified_recordings(
            input_root / "dataset",
            teachers,
            manifest_sha256=configuration.dataset.manifest_sha256,
            index_sha256=configuration.dataset.shards_sha256,
            event=context.session.event,
        )
        ranges = [evaluation.development_seed, evaluation.final_seed]
        ranges.extend(
            phase.heldout_seed for phase in configuration.phases if phase.heldout_seed is not None
        )
        collection_end = evidence["collection_seed"] + max(
            info["episodes"] for info in evidence["collection"].values()
        )
        if evidence["lanes"] != evaluation.lanes or min(ranges) < collection_end:
            raise ValueError(
                "frozen recordings require matching vector lanes and disjoint evaluation seeds"
            )
    else:
        collection = configuration.collection
        assert collection is not None
        destination = context.run_dir / "distillation-dataset"
        destination.mkdir()
        shards, evidence = collect_recordings(
            destination,
            teachers,
            episodes=collection.attempts_per_task,
            lanes=evaluation.lanes,
            seed=collection.seed_start,
            event=context.session.event,
        )
    context.session.configure_budget(
        requested_limit=configuration.total_updates,
        step_quantum=1,
        progress_fields=(ProgressField(TRAIN_DISTILL_CROSS_ENTROPY, "imitation loss"),),
    )
    actions = tuple(dict.fromkeys(action for teacher in teachers for action in teacher.actions))
    if tuple(evidence["actions"]) != actions:
        raise ValueError("recorded action union differs from the pinned teacher contract")
    columns = action_columns(teachers, actions)
    torch.manual_seed(configuration.phases[0].seed)
    policy = DistilledPolicy(
        image_shape=evidence["image_shape"],
        actions=actions,
        tasks=[(teacher.level, ids) for teacher, ids in zip(teachers, columns, strict=True)],
        temperature=evaluation.development_temperature,
        device=device,
    )
    context.mark_ready()

    def assess_student(seed: int, episodes: int, temperature: float):
        return assess(
            teachers,
            policy.actor,
            episodes=episodes,
            lanes=evaluation.lanes,
            seed=seed,
            device=device,
            temperature=temperature,
            columns=columns,
        )

    def assess_teacher(seed: int, episodes: int):
        return assess(
            teachers, None, episodes=episodes, lanes=evaluation.lanes, seed=seed, device=device
        )

    def comparison(seed: int, temperature: float) -> dict[str, Any]:
        teacher = assess_teacher(seed, evaluation.heldout_episodes)
        student = assess_student(seed, evaluation.heldout_episodes, temperature)
        result = {
            "seed": seed,
            "temperature": temperature,
            "episodes_per_task": evaluation.heldout_episodes,
            "teacher": counts(teacher),
            "student": counts(student),
            "teacher_panel": teacher,
            "student_panel": student,
        }
        context.session.event(f"distillation held-out panel {seed}: {result['student']}")
        return result

    dev_teacher = counts(
        assess_teacher(evaluation.development_seed, evaluation.development_episodes)
    )
    report: dict[str, Any] = {
        "dataset_examples": sum(
            info["retained_transitions"] for info in evidence["collection"].values()
        ),
        "development_teacher": dev_teacher,
        "phases": [],
    }
    total_step = 0
    selected_updates = 0
    for phase in configuration.phases:
        if context.stop_flag.requested:
            break
        completed, development, best_at = _fit_phase(
            context,
            configuration,
            phase,
            policy.actor,
            shards,
            columns,
            len(actions),
            assess_student,
            dev_teacher,
            continuation=total_step > 0,
            step=total_step,
            device=device,
        )
        total_step += completed
        selected_updates += best_at
        phase_report = {
            "completed_updates": completed,
            "selected_updates": best_at,
            "development": development,
        }
        if phase.heldout_seed is not None and not context.stop_flag.requested:
            phase_report["heldout"] = comparison(
                phase.heldout_seed, evaluation.development_temperature
            )
        report["phases"].append(phase_report)

    calibration = []
    for temperature in evaluation.temperatures:
        if context.stop_flag.requested:
            break
        clear_counts = counts(
            assess_student(
                evaluation.development_seed, evaluation.development_episodes, temperature
            )
        )
        calibration.append({"temperature": temperature, "clears": clear_counts})
    if calibration:
        best = max(calibration, key=lambda row: _score(dev_teacher, row["clears"]))
        policy.temperature = best["temperature"]
    report.update(
        calibration=calibration,
        selected_updates=selected_updates,
        selected_temperature=policy.temperature,
    )
    if not context.stop_flag.requested:
        report["final"] = comparison(evaluation.final_seed, policy.temperature)
    atomic_write_json(context.run_dir / "distillation-results.json", report)

    reason = context.session.terminal_reason(TerminalReason.RESOURCE_EXHAUSTION)
    kind = context.session.terminal_model_kind(reason)
    context.train_config["training_terminal"] = context.session.terminal_provenance(
        terminal_reason=reason,
        final_step=total_step,
    )
    active_env = make_eval_vec_env(
        context.environment, n_envs=1, seed=0, rom_binding=context.rom_binding
    )
    try:
        action_contract = runtime_action_contract(active_env)
        policy.bind_action_contract(action_contract)
        policy.bind_action_space(active_env.action_space)
    finally:
        active_env.close()
    context.session.checkpoints.save(
        kind=kind,
        step=total_step,
        model_path=context.run_dir / "final_model.zip",
        terminal=True,
        save_bundle=lambda path, artifact_kind, artifact_step: install_model_bundle(
            path,
            save_checkpoint=lambda destination: policy.save(destination, report=report),
            train_config=context.train_config,
            config=context.environment,
            kind=artifact_kind,
            checkpoint_step_value=artifact_step,
            action_contract=action_contract,
        ),
    )
    return context.session.result(terminal_reason=reason, final_step=total_step, model_kind=kind)


class DistillBackend:
    backend_id = BACKEND_ID

    def normalize_config(self, config, *, label):
        try:
            return DistillationConfig.model_validate(config).model_dump(mode="json")
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc

    def validate(self, common_config, backend_config):
        configuration = DistillationConfig.model_validate(backend_config)
        if common_config.get("checkpoint_eval_backend") != "none":
            raise ValueError(
                "distillation uses its explicit comparison protocol and requires checkpoint_eval_backend=none"
            )
        if common_config.get("policy_model") is not None:
            raise ValueError("distillation selects the student architecture in its own recipe")
        timesteps = common_config.get("timesteps")
        if timesteps is not None and timesteps != configuration.total_updates:
            raise ValueError(
                "distillation timesteps must equal the sum of phase optimizer-update limits"
            )

    def run(self, context):
        return run_distillation(context, DistillationConfig.model_validate(context.backend_config))

    def acceptance_mode(self, backend_config):
        return CHECKPOINT_EVAL_ACCEPTANCE

    def contract_payload(self):
        return {
            "schema_version": 1,
            "status": "available",
            "schema": DistillationConfig.model_json_schema(),
        }

    def state_archive_priority_metrics(self):
        return ()

    def runtime_metadata(self, backend_config):
        return {
            "training_backend_id": self.backend_id,
            "algorithm_id": "distill",
            "model_class": STUDENT_MODEL_CLASS,
        }


BACKENDS = {BACKEND_ID: DistillBackend()}
