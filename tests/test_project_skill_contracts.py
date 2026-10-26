from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_SKILL = REPO_ROOT / ".codex/skills/gradlab_launch-experiment/SKILL.md"


def test_launch_skill_describes_goal_owned_recipe_location() -> None:
    skill = LAUNCH_SKILL.read_text()
    assert "(references/operations.md)" in skill
    operations = (LAUNCH_SKILL.parent / "references/operations.md").read_text()
    assert "one recipe under `experiments/recipes/`" not in skill + operations
    assert "selected goal's `recipes/` directory" in operations
    assert "`experiments/recipes/_presets/`" in operations
