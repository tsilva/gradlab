from __future__ import annotations

import unittest
from types import SimpleNamespace

from gradlab.metric_names import (
    EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN,
    EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN,
    LEADER_CHECKPOINT_ARTIFACT_REF,
    LEADER_CHECKPOINT_EVALUATION_SOURCE,
    LEADER_CHECKPOINT_PROJECTION_TIMESTAMP,
    LEADER_CHECKPOINT_RETURN_SHAPED_MEAN,
    LEADER_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_OUTCOME_SUCCESS_STARTS_RATE_MIN,
    METRICS_SCHEMA_VERSION,
)
from gradlab.wandb_leaders import (
    CheckpointLeader,
    RunScore,
    checkpoint_leader as parse_checkpoint_leader,
    checkpoint_summary_filter,
    rank_checkpoint_leaders,
    rank_run_leaders,
    run_score as score_wandb_run,
)


def run_score(
    *,
    reward_shape: str,
    reward_hash: str,
    effective_goal_hash: str,
    objective: float,
    is_default: bool,
    recipe_slug: str = "ppo",
    steps: int | None = None,
) -> RunScore:
    return RunScore(
        goal_slug="mario-level-1-1",
        recipe_slug=recipe_slug,
        reward_shape=reward_shape,
        reward_shape_sha256=reward_hash,
        effective_goal_contract_sha256=effective_goal_hash,
        reward_shape_is_default=is_default,
        run_id=f"run-{reward_shape}-{objective}",
        run_name=f"run-{reward_shape}",
        url="https://example.invalid/run",
        seed=1,
        objective=objective,
        steps=steps,
    )


def checkpoint_leader(
    *,
    reward_shape: str,
    reward_hash: str,
    effective_goal_hash: str,
    rank_score: tuple[float, ...],
    is_default: bool,
) -> CheckpointLeader:
    return CheckpointLeader(
        goal_slug="mario-level-1-1",
        recipe_slug="ppo",
        reward_shape=reward_shape,
        reward_shape_sha256=reward_hash,
        effective_goal_contract_sha256=effective_goal_hash,
        reward_shape_is_default=is_default,
        run_id=f"run-{reward_shape}-{rank_score[0]}",
        run_name=f"run-{reward_shape}",
        url="https://example.invalid/run",
        objective=rank_score[0],
        objective_name="success",
        rank_values={"success": rank_score[0]},
        checkpoint_step=1,
        artifact_ref="model:v1",
        eval_source="acceptance",
        projection_updated_at="2026-08-07T00:00:00Z",
        rank_score=rank_score,
    )


class WandbLeaderRewardShapeTests(unittest.TestCase):
    def test_training_only_run_uses_configured_final_return_and_steps(self):
        score = score_wandb_run(
            SimpleNamespace(
                config={
                    "goal_slug": "Breakout-Atari2600-v0",
                    "recipe_slug": "ppo",
                    "selection_rank": [
                        "max(train/episode/return/shaped/origin/target/rolling/mean)",
                        "min(train/global_step)",
                    ],
                },
                summary={
                    "train/episode/return/shaped/origin/target/rolling/mean": 42.0,
                    "train/outcome/success/starts/all/rolling/rate/min": 0.1,
                    "train/global_step": 1_000_000,
                },
                tags=(),
                id="run",
                name="run",
                url="https://example.invalid/run",
            ),
            objective_keys=("train/outcome/success/starts/all/rolling/rate/min",),
        )

        self.assertIsNotNone(score)
        assert score is not None
        self.assertEqual(score.objective, 42.0)
        self.assertEqual(score.steps, 1_000_000)

    def test_checkpoint_query_filters_only_for_artifact_presence(self):
        self.assertEqual(
            checkpoint_summary_filter(),
            {f"summary_metrics.{LEADER_CHECKPOINT_ARTIFACT_REF}": {"$exists": True}},
        )

    def test_checkpoint_leader_unwraps_reducers_and_uses_dynamic_rank_values(self):
        leader = parse_checkpoint_leader(
            SimpleNamespace(
                config={
                    "metrics_schema_version": METRICS_SCHEMA_VERSION,
                    "goal_slug": "mario-level-1-1",
                    "recipe_slug": "ppo",
                    "selection_rank": [
                        f"max({EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN})",
                        f"max({EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN})",
                        f"min({LEADER_CHECKPOINT_STEP})",
                    ],
                },
                summary={
                    LEADER_CHECKPOINT_ARTIFACT_REF: "model:v1",
                    LEADER_CHECKPOINT_EVALUATION_SOURCE: "modal:automatic",
                    LEADER_CHECKPOINT_PROJECTION_TIMESTAMP: "2026-08-07T00:00:00Z",
                    LEADER_CHECKPOINT_STEP: {"min": 200},
                    LEADER_CHECKPOINT_OUTCOME_SUCCESS_STARTS_RATE_MIN: {"max": 0.75},
                    LEADER_CHECKPOINT_RETURN_SHAPED_MEAN: {"last": 8.0},
                },
                id="run",
                name="run",
                url="https://example.invalid/run",
            )
        )

        self.assertIsNotNone(leader)
        assert leader is not None
        self.assertEqual(
            dict(leader.rank_values),
            {
                EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN: 0.75,
                EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN: 8.0,
                LEADER_CHECKPOINT_STEP: 200.0,
            },
        )
        self.assertEqual(leader.checkpoint_step, 200)

    def test_checkpoint_leader_rejects_missing_configured_rank_projection(self):
        leader = parse_checkpoint_leader(
            SimpleNamespace(
                config={
                    "metrics_schema_version": METRICS_SCHEMA_VERSION,
                    "selection_rank": [
                        f"max({EVAL_FULL_OUTCOME_SUCCESS_STARTS_RATE_MIN})",
                        f"max({EVAL_FULL_EPISODE_RETURN_SHAPED_MEAN})",
                    ],
                },
                summary={
                    LEADER_CHECKPOINT_ARTIFACT_REF: "model:v1",
                    LEADER_CHECKPOINT_EVALUATION_SOURCE: "modal:automatic",
                    LEADER_CHECKPOINT_PROJECTION_TIMESTAMP: "2026-08-07T00:00:00Z",
                    LEADER_CHECKPOINT_STEP: 200,
                    LEADER_CHECKPOINT_OUTCOME_SUCCESS_STARTS_RATE_MIN: 0.75,
                },
            )
        )

        self.assertIsNone(leader)

    def test_run_ranking_does_not_combine_different_reward_shapes(self):
        leaders = rank_run_leaders(
            [
                run_score(
                    reward_shape="score-v1",
                    reward_hash="sha256:score",
                    effective_goal_hash="sha256:goal-score",
                    objective=0.4,
                    is_default=True,
                ),
                run_score(
                    reward_shape="score-step-0p01-v1",
                    reward_hash="sha256:step",
                    effective_goal_hash="sha256:goal-step",
                    objective=0.9,
                    is_default=False,
                ),
            ]
        )

        self.assertEqual(len(leaders), 2)
        self.assertEqual(leaders[0].reward_shape, "score-v1")
        self.assertEqual({leader.seeds for leader in leaders}, {1})

    def test_checkpoint_ranking_is_scoped_to_each_reward_shape(self):
        leaders = rank_checkpoint_leaders(
            [
                checkpoint_leader(
                    reward_shape="score-v1",
                    reward_hash="sha256:score",
                    effective_goal_hash="sha256:goal-score",
                    rank_score=(0.4,),
                    is_default=True,
                ),
                checkpoint_leader(
                    reward_shape="score-step-0p01-v1",
                    reward_hash="sha256:step",
                    effective_goal_hash="sha256:goal-step",
                    rank_score=(0.9,),
                    is_default=False,
                ),
            ]
        )

        self.assertEqual(
            [leader.reward_shape for leader in leaders],
            ["score-v1", "score-step-0p01-v1"],
        )

    def test_training_return_tie_prefers_fewer_mean_policy_transitions(self):
        common = {
            "reward_shape": "score-v1",
            "reward_hash": "sha256:score",
            "effective_goal_hash": "sha256:goal-score",
            "is_default": True,
        }
        leaders = rank_run_leaders(
            [
                run_score(**common, recipe_slug="fast", objective=10.0, steps=100),
                run_score(**common, recipe_slug="fast", objective=20.0, steps=200),
                run_score(**common, recipe_slug="slow", objective=10.0, steps=300),
                run_score(**common, recipe_slug="slow", objective=20.0, steps=400),
            ],
            min_seeds=2,
        )

        self.assertEqual([leader.recipe_slug for leader in leaders], ["fast", "slow"])
        self.assertEqual(leaders[0].mean_steps, 150)
        self.assertEqual(
            (leaders[0].worst_seed, leaders[0].mean_seed, leaders[0].best_seed),
            (10.0, 15.0, 20.0),
        )


if __name__ == "__main__":
    unittest.main()
