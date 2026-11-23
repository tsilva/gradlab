from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Discrete

from gradlab.distillation import balanced_batch, verified_recordings, write_recording_index
from gradlab.distillation_policy import DistilledPolicy
from gradlab.file_utils import atomic_write_json, file_sha256
from gradlab.policy_runtime import PolicyRuntime
from gradlab.recipe_documents import compose_train_document


def _action_contract(*names):
    return {
        "policy": {
            "space": {"type": "discrete", "start": 0, "n": len(names)},
            "semantics": {
                "encoding": "explicit",
                "status": "available",
                "entries": [{"value": i, "semantic_id": name} for i, name in enumerate(names)],
            },
        },
    }


def test_distilled_policy_round_trip_maps_task_actions_by_meaning(tmp_path):
    policy = DistilledPolicy(
        image_shape=(4, 84, 84),
        actions=("wait", "move", "use"),
        tasks=(("RoomA", (0, 1)), ("RoomB", (1, 2))),
        temperature=0.5,
        device="cpu",
    )
    with torch.no_grad():
        for tensor in policy.actor.parameters():
            tensor.zero_()
        policy.actor.actor[-1].bias.copy_(torch.tensor([0.0, 1.0, 2.0]))
    path = tmp_path / "student.zip"
    policy.save(path)
    replay = PolicyRuntime(DistilledPolicy.load(path, device="cpu"), algorithm_id="distill")
    replay.bind_action_space(Discrete(3), _action_contract("use", "wait", "move"))
    obs = np.zeros((2, 4, 84, 84), np.uint8)
    result = replay.decide(
        obs,
        action_selection_mode="deterministic",
        execution_context=("RoomA", "RoomB"),
    )
    assert result.actions.tolist() == [2, 0]
    assert result.decisions[0].probabilities[0] == 0
    assert result.decisions[1].probabilities[1] == 0
    with pytest.raises(ValueError, match="no task"):
        replay.decide(obs, execution_context=("RoomA", "Unknown"))


def test_balanced_dataset_checks_bytes_and_maps_native_probabilities(tmp_path):
    teachers = [
        SimpleNamespace(level="RoomA", identity={"a": 1}, actions=("wait", "move")),
        SimpleNamespace(level="RoomB", identity={"b": 1}, actions=("move", "use")),
    ]
    names = [teacher.level for teacher in teachers]
    evidence = {
        "teachers": {teacher.level: teacher.identity for teacher in teachers},
        "levels": names,
        "actions": ["wait", "move", "use"],
        "level_actions": {teacher.level: teacher.actions for teacher in teachers},
        "image_shape": [4, 84, 84],
        "collection": {
            teacher.level: {"retained_transitions": index + 1}
            for index, teacher in enumerate(teachers)
        },
    }
    atomic_write_json(tmp_path / "dataset.json", evidence)
    for index, teacher in enumerate(teachers):
        folder = tmp_path / teacher.level
        folder.mkdir()
        np.full((index + 1, 4, 84, 84), index, np.uint8).tofile(folder / "images.bin")
        np.tile(np.array([0.25, 0.75], dtype=np.float16), (index + 1, 1)).tofile(
            folder / "probabilities.bin",
        )
        atomic_write_json(folder / "episodes.json", [])
    write_recording_index(tmp_path, names)
    digests = {
        "manifest_sha256": file_sha256(tmp_path / "dataset.json"),
        "index_sha256": file_sha256(tmp_path / "shards.json"),
        "event": lambda message: None,
    }
    shards, _ = verified_recordings(tmp_path, teachers, **digests)
    _, ids, targets = balanced_batch(
        shards,
        2,
        np.random.default_rng(1),
        "cpu",
        columns=[(0, 1), (1, 2)],
        action_count=3,
    )
    assert ids.tolist() == [0, 0, 1, 1]
    assert targets.tolist() == [[0.25, 0.75, 0]] * 2 + [[0, 0.25, 0.75]] * 2
    (tmp_path / "RoomA/probabilities.bin").write_bytes(bytes(4))
    with pytest.raises(ValueError, match="shard has changed"):
        verified_recordings(tmp_path, teachers, **digests)


def test_mario_reproduction_recipe_compiles_with_separate_seed_panels():
    goal = Path("experiments/goals/SuperMarioBros-Nes-v0/DistillAllLevels")
    config = compose_train_document(goal / "_goal.yaml", goal / "recipes/offline-distill.yaml")
    train = config["train_config"]
    backend = train["training_backend"]
    assert backend["id"] == "gradlab.distill"
    assert train["checkpoint_eval_backend"] == "none"
    assert train["timesteps"] == 72_000
    assert len(backend["config"]["teachers"]) == 32
    assert [phase["heldout_seed"] for phase in backend["config"]["phases"]] == [600000, 700000]
    assert backend["config"]["evaluation"]["final_seed"] == 800000
