"""Isolated Mario goal trees for portable backend contract tests."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile


@contextmanager
def mario_recipe_fixture(kind):
    fixture, filename = {
        "go-explore": ("mario-go-explore-route.yaml", "go-explore-route-diagnostic.yaml"),
        "critic-warmup": ("mario-critic-warmup.yaml", "ppo-sft-critic-warmup.yaml"),
    }[kind]
    checkout = Path(__file__).resolve().parents[1]
    family = Path("experiments/goals/SuperMarioBros-Nes-v0")
    with tempfile.TemporaryDirectory(prefix="gradlab-mario-contract-") as temporary:
        root = Path(temporary)
        for relative in (
            family / "Level1-3/_goal.yaml", family / "_goal-common.yaml",
            family / "_goal-family-common.yaml", family / "_reward-shapes-common.yaml",
            Path("experiments/tracking.yaml"),
        ):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkout / relative, target)
        presets = Path("experiments/recipes/_presets")
        shutil.copytree(checkout / presets, root / presets)
        goal = root / family / "Level1-3/_goal.yaml"
        recipe = goal.parent / "recipes" / filename
        recipe.parent.mkdir()
        shutil.copyfile(checkout / "tests/fixtures" / fixture, recipe)
        yield root, goal, recipe
