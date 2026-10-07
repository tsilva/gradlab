"""Bounded representative inventory and intended curriculum start distributions."""

from collections.abc import Mapping

TABLE = "train/curriculum/distribution"
COLUMNS = (
    "cell",
    "representatives",
    "probability",
    "cold",
    "strategy",
    "rollout",
    "archive_lanes",
    "normal_lanes",
    "recent_combined_count",
)


def validate_distribution(value):
    if not isinstance(value, Mapping) or set(value) != {"rows"}:
        raise ValueError("invalid curriculum distribution")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) > 16384:
        raise ValueError("curriculum distribution exceeds its bounded inventory")
    cells = set()
    for row in rows:
        if not isinstance(row, list) or len(row) != len(COLUMNS):
            raise ValueError("invalid curriculum distribution row")
        cell, entries, probability, cold, strategy, rollout, archive, normal, count = row
        if (
            not isinstance(cell, str)
            or not cell
            or cell in cells
            or len(cell) > 1024
            or any(
                type(item) is not int or item < 0 for item in (entries, rollout, archive, normal)
            )
            or entries == 0
            or type(cold) is not bool
            or strategy not in {"coverage", "value_error"}
            or (
                probability is not None
                and (type(probability) is not float or not 0 <= probability <= 1)
            )
            or (count is not None and (type(count) is not int or count < 0))
        ):
            raise ValueError("invalid curriculum distribution values")
        cells.add(cell)
    return {"rows": [list(row) for row in rows]}


def distribution_table(payload, *, run_id):
    import wandb

    if not isinstance(run_id, str) or not run_id:
        raise ValueError("curriculum table requires its publishing Run identity")
    return wandb.Table(
        columns=[*COLUMNS, "run_id"],
        data=[[*row, run_id] for row in validate_distribution(payload)["rows"]],
    )
