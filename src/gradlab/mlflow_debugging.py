"""Portable chart guidance for the native MLflow workspace."""

DEBUGGING_DASHBOARD = {
    "schema_version": 1,
    "section": "Debugging",
    "line_smoothing": 0,
    "ignore_outliers": False,
    "charts": [
        {"metric": metric, "axis": "wall_time" if metric.startswith("ops/") else "step"}
        for metric in (
            "train/success/min", "train/return/mean", "train/kl/mean",
            "train/clip/fraction", "train/entropy/mean", "train/value_loss/mean",
            "train/explained_variance", "train/throughput/rate",
            "train/between_rollouts/seconds", "ops/outbox/count", "ops/visibility/seconds",
        )
    ],
    "hardware": {"tab": "System metrics", "axis": "wall_time", "prefix": "system/"},
    "missing_evidence": "Omit unsupported metrics; never substitute zero or current host readings.",
    "instructions": "Add one line chart per metric in Model metrics; collapse automatic groups. "
                    "Native workspace layout is saved in the browser. This file is portable guidance.",
}
