import csv
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx2
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from gradlab.dashboard_publication import _MARIO_LEVELS, publish_game_dashboard


OWNER = "example"
MARIO = "SuperMarioBros-Nes-v0"
RELEASE = "a" * 40
SPACE_PARENT = "b" * 40
SPACE_COMMIT = "c" * 40
PROFILE = "d" * 64


def release_for(environment, goal):
    return {
        "repository": {
            "canonical_environment_id": environment,
            "goal_id": goal,
            "repo_id": f"{OWNER}/{environment}_{goal}" if environment != goal else f"{OWNER}/{environment}",
        },
        "release": {"version": "v2"},
        "lineage": {"algorithm": "ppo"},
        "replay": {"execution": {"provider_id": "fake-provider" if environment != MARIO else "env-supermariobrosnes-turbo-emu"}},
    }


EVIDENCE = {
    "identity": {"checkpoint_sha256": PROFILE, "checkpoint_step": 1000},
    "protocol": {"seed": 50000},
    "acceptance": {
        "passed": True,
        "outcomes": [{
            "metric": "eval/success/mean", "label": "Success rate", "unit": "fraction",
            "value": 0.75, "operator": ">=",
        }],
    },
    "episode_results": [
        {"outcome": "success", "steps": 20},
        {"outcome": "failure", "steps": 14, "events": ["life_loss"]},
    ],
}
RECIPE = {"recipe": {"train_config": {"training_backend": {"config": {"sil": {"enabled": True}}}}}}


class FakeSpace:
    def __init__(self, tmp_path, environment, files):
        self.tmp_path = tmp_path
        self.repo = f"{OWNER}/{environment}_Dashboard"
        self.files = files
        self.sha = SPACE_PARENT
        self.api = Mock()
        self.api.space_info.side_effect = self.info
        self.api.create_repo.side_effect = self.create
        self.api.create_commit.side_effect = self.commit
        self.model = SimpleNamespace(item_type="model", item_id=f"{OWNER}/model")
        self.dashboard = SimpleNamespace(item_type="space", item_id=self.repo, item_object_id="space-id")
        self.items = [self.model]
        self.api.get_collection.side_effect = lambda *_args, **_kwargs: SimpleNamespace(
            owner={"name": OWNER}, private=False, items=list(self.items),
        )
        self.api.add_collection_item.side_effect = lambda *_args, **_kwargs: self.items.append(self.dashboard)
        self.api.update_collection_item.side_effect = self.move_first

    def info(self, repo, **_kwargs):
        assert repo == self.repo
        if self.files is None:
            request = httpx2.Request("GET", f"https://huggingface.co/api/spaces/{repo}")
            raise RepositoryNotFoundError("missing", response=httpx2.Response(401, request=request))
        return SimpleNamespace(
            id=repo, sha=self.sha, sdk="static", private=False, gated=False,
            siblings=[SimpleNamespace(rfilename=name, size=len(value.encode())) for name, value in self.files.items()],
        )

    def create(self, repo, **_kwargs):
        assert repo == self.repo and self.files is None
        self.files = {".gitattributes": ""}

    def commit(self, repo, *, parent_commit, operations, **_kwargs):
        assert repo == self.repo and parent_commit == self.sha
        self.files.update({op.path_in_repo: op.path_or_fileobj.decode() for op in operations})
        self.sha = SPACE_COMMIT
        return SimpleNamespace(oid=self.sha)

    def download(self, repo, name, *, revision, repo_type, token):
        assert repo == self.repo and repo_type == "space" and revision == self.sha
        path = self.tmp_path / name
        path.write_bytes(self.files[name].encode())
        return str(path)

    def move_first(self, slug, item, *, position):
        assert item == "space-id" and position == 0
        self.items = [self.dashboard, self.model]

    def publish(self, manifest):
        return publish_game_dashboard(
            self.api, self.download, token="fixture", collection_slug=f"{OWNER}/collection-id",
            manifest=manifest, evidence=EVIDENCE, recipe=RECIPE, commit=RELEASE,
        )


def test_create_and_reconcile_dashboard_after_lost_collection_ack(tmp_path):
    space = FakeSpace(tmp_path, "VizdoomBasic-v1", None)
    manifest = release_for("VizdoomBasic-v1", "VizdoomBasic-v1")
    space.api.update_collection_item.side_effect = OSError("simulated interrupted update")
    with pytest.raises(OSError):
        space.publish(manifest)

    space.api.update_collection_item.side_effect = space.move_first
    assert space.publish(manifest) == f"https://huggingface.co/spaces/{space.repo}"
    space.api.create_commit.assert_called_once()
    space.api.add_collection_item.assert_called_once()
    assert space.items == [space.dashboard, space.model]
    stage, = json.loads(space.files["dashboard.json"])["stages"]
    assert stage["model_url"].endswith("/tree/" + RELEASE)
    assert stage["replay_url"].endswith("/resolve/" + RELEASE + "/replay.mp4")
    assert stage["recipe_url"].endswith("/blob/" + RELEASE + "/recipe.json")
    assert stage["metric"] == {"value": 0.75, "display": "75.0%"}
    assert stage["algorithm"] == "PPO + SIL"
    assert 'data-sort="trainingSteps"' in space.files["index.html"]
    assert 'id="replay-position"' in space.files["index.html"]
    checkout = Path(__file__).resolve().parents[1]
    script = checkout / ".codex/skills/gradlab_upload-checkpoint/scripts/render_dashboard.py"
    offline = tmp_path / "offline"
    subprocess.run([sys.executable, str(script), str(tmp_path / "dashboard.json"), str(offline)], check=True)
    assert (offline / "index.html").read_text() == space.files["index.html"]


def legacy_mario_files():
    original = "e" * 40
    csv_output = io.StringIO(newline="")
    fields = [
        "level", "status", "checkpoint_status", "model_url", "model_card_url", "replay_url",
        "checkpoint_sha256", "training_steps", "evaluation_seeds", "total_clears", "total_episodes",
        "clear_rate_percent", "clear_rate_sample_stddev_pp", "clear_steps_mean",
        "clear_steps_sample_stddev", "seed_20000_clears", "failed_episodes", "failure_life_loss",
    ]
    writer = csv.DictWriter(csv_output, fieldnames=fields)
    writer.writeheader()
    rows = []
    for level in _MARIO_LEVELS:
        if level not in ("1-1", "1-2"):
            writer.writerow({"level": level, "status": "Unsolved"})
            rows.append(f'<tr><td aria-label="Unsolved"></td><th scope="row">World {level}</th><td></td></tr>')
            continue
        base = f"https://huggingface.co/{OWNER}/{MARIO}_Level{level}"
        card = f"{base}/blob/{original}/README.md"
        writer.writerow({
            "level": level, "status": "Solved", "checkpoint_status": "Published",
            "model_url": f"{base}/tree/{original}", "model_card_url": card,
            "replay_url": f"{base}/resolve/{original}/replay.mp4",
            "checkpoint_sha256": "f" * 64, "training_steps": 700,
            "evaluation_seeds": 3, "total_clears": 299, "total_episodes": 300,
            "clear_rate_percent": 99.7, "clear_rate_sample_stddev_pp": 0.6,
            "clear_steps_mean": 15, "clear_steps_sample_stddev": 1, "seed_20000_clears": 99,
            "failed_episodes": 1, "failure_life_loss": 1,
        })
        recipe = f"https://github.com/tsilva/gradlab/blob/{original}/experiments/goal/ppo.yaml"
        rows.append(
            f'<tr><td aria-label="Solved"></td><th scope="row">World {level}</th><td>PPO</td>'
            f'<td><a href="{card}">Model card</a><a href="{recipe}">Recipe</a></td></tr>'
        )
    return {
        "README.md": "---\nsdk: static\napp_file: index.html\n---\n",
        "style.css": "unrelated.css",
        "index.html": '<p id="results-method">A stage is solved after one clear.</p>\n' + "\n".join(rows),
        "results.csv": csv_output.getvalue(),
    }


@pytest.mark.parametrize("same_checkpoint", [False, True])
def test_refresh_mario_preserves_stage_order_recipes_and_other_csv_rows(tmp_path, same_checkpoint):
    original = legacy_mario_files()
    if same_checkpoint:
        original["results.csv"] = original["results.csv"].replace("f" * 64, PROFILE)
    space = FakeSpace(tmp_path, MARIO, dict(original))
    manifest = release_for(MARIO, "Level1-2")

    space.publish(manifest)
    space.publish(manifest)

    data = json.loads(space.files["dashboard.json"])
    assert [row["id"] for row in data["stages"]] == [f"Level{level}" for level in _MARIO_LEVELS]
    assert data["stages"][0]["metric"]["display"] == "299 / 300"
    updated = data["stages"][1]
    if same_checkpoint:
        assert updated["metric"]["display"] == "299 / 300"
        assert updated["evaluation_seeds"] == 3
    else:
        assert updated["metric"] == {"value": 0.5, "display": "1 / 2", "detail": "50.0%"}
        assert updated["clear_steps"] == {"value": 20, "display": "20.0"}
    assert updated["current_recipe_url"].endswith("/experiments/goal/ppo.yaml")
    assert data["stages"][-1] == {"id": "Level8-4", "label": "World 8-4", "solved": False}
    updated_csv = list(csv.DictReader(io.StringIO(space.files["results.csv"])))
    original_csv = list(csv.DictReader(io.StringIO(original["results.csv"])))
    assert updated_csv[0] == original_csv[0] and updated_csv[2:] == original_csv[2:]
    assert updated_csv[1]["seed_20000_clears"] == ("99" if same_checkpoint else "")
    assert updated_csv[1]["failed_episodes"] == updated_csv[1]["failure_life_loss"] == "1"
    assert space.files["style.css"] == original["style.css"]
    space.api.create_commit.assert_called_once()
