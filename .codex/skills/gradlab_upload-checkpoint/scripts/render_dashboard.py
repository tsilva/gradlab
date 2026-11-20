"""Render a credential-free, static GradLab dashboard from reviewed public data."""

import argparse
import json
import math
import re
from html import escape
from pathlib import Path
from string import Template
from urllib.parse import urlsplit


ASSET = Path(__file__).resolve().parents[1] / "assets" / "dashboard.html"
ABSENT = '<td class="absent" aria-label="Unavailable">—</td>'


def check_fields(value, required, optional=()):
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"Missing fields {sorted(missing)}, unexpected fields {sorted(extra)}")


def linked(label, url, accessible_name=None):
    if not url:
        return escape(label)
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query:
        raise ValueError("Links must be public HTTPS URLs without credentials or query parameters")
    aria = f' aria-label="{escape(accessible_name)}"' if accessible_name else ""
    return f'<a href="{escape(url)}" target="_blank" rel="noopener"{aria}>{escape(label)}</a>'


def numeric_cell(metric, css_class=""):
    if metric is None:
        return ABSENT
    check_fields(metric, ("value", "display"), ("detail",))
    value = metric["value"]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Sort values must be finite numbers")
    detail = f'<span class="rate">{escape(metric["detail"])}</span>' if metric.get("detail") else ""
    return f'<td class="num"><span class="{css_class}">{escape(metric["display"])}</span>{detail}</td>'


def heading(label, key, direction="ascending"):
    attributes = ' aria-sort="ascending"' if key == "stage" else ' class="num"'
    return (
        f'<th scope="col"{attributes}><button class="sort-button" type="button" data-sort="{key}" '
        f'data-initial-direction="{direction}">{escape(label)}'
        '<span class="sort-icon" aria-hidden="true"></span></button></th>'
    )


def render_row(stage, show_solved, show_clear_steps):
    check_fields(
        stage, ("id", "label"),
        ("algorithm", "solved", "metric", "training_steps", "clear_steps", "evaluation_seeds",
         "model_url", "checkpoint_sha256", "recipe_url", "current_recipe_url", "replay_url"),
    )
    label = stage["label"]
    cells = []
    if show_solved:
        status = {True: ("Solved", "✅"), False: ("Unsolved", "❌")}.get(stage.get("solved"))
        cells.append(
            f'<td class="solved"><span role="img" aria-label="{status[0]}">{status[1]}</span></td>'
            if status else ABSENT
        )
    cells.append(f'<th scope="row">{escape(label)}</th>')
    algorithm = stage.get("algorithm")
    cells.append(f"<td>{escape(algorithm)}</td>" if algorithm else ABSENT)
    metric = stage.get("metric")
    cells.append(numeric_cell(metric, "score"))
    steps = stage.get("training_steps")
    if steps is not None and (type(steps) is not int or steps < 0):
        raise ValueError("Training steps must be a nonnegative integer")
    cells.append(numeric_cell({"value": steps, "display": f"{steps:,}"} if steps is not None else None))
    clear_steps = stage.get("clear_steps")
    if show_clear_steps:
        cells.append(numeric_cell(clear_steps))
    clear_steps = clear_steps if show_clear_steps else None

    attributes = []
    for key, value in (
        ("metric", metric["value"] if metric is not None else None),
        ("training-steps", steps),
        ("clear-steps", clear_steps["value"] if clear_steps is not None else None),
    ):
        if value is not None:
            attributes.append(f' data-{key}="{value}"')

    model = stage.get("model_url")
    recipe = stage.get("recipe_url")
    current_recipe = stage.get("current_recipe_url")
    replay = stage.get("replay_url")
    if model:
        if not re.fullmatch(r"[a-f0-9]{64}", stage.get("checkpoint_sha256", "")):
            raise ValueError("A published model needs its checkpoint SHA-256")
        match = re.fullmatch(r"https://huggingface\.co/([^/]+/[^/]+)/tree/([a-f0-9]{40})", model)
        if not match:
            raise ValueError("Model links must pin an immutable Hugging Face commit")
        model_repo, revision = match.groups()
        if replay:
            if replay != f"https://huggingface.co/{model_repo}/resolve/{revision}/replay.mp4":
                raise ValueError("Replay and model must identify the same Hub revision")
            attributes.extend((f' data-replay-label="{escape(label)}"', f' data-replay-url="{escape(replay)}"'))
    elif recipe or current_recipe or replay or algorithm or metric is not None or steps is not None or clear_steps is not None:
        raise ValueError("Public evidence must identify its published model")
    links = []
    if model:
        card = f"https://huggingface.co/{model_repo}/blob/{revision}/README.md"
        links.append(linked("Model card", card, f"Model card for {label}"))
    if current_recipe:
        links.append(linked("Recipe", current_recipe, f"Current recipe for {label}"))
    elif recipe:
        links.append(linked("Recipe", recipe, f"Source recipe for {label}"))
    cells.append(f'<td class="links">{"".join(links)}</td>' if links else ABSENT)
    return f'<tr{"".join(attributes)}>{"".join(cells)}</tr>'


def render(data):
    check_fields(
        data, ("version", "environment", "provider", "provider_url", "collection_url",
               "metric_name", "metric_column", "metric_direction", "methods", "stages"),
        ("clear_steps",),
    )
    if data["version"] != 1 or data["metric_direction"] not in ("ascending", "descending"):
        raise ValueError("Expected dashboard version 1 and a valid metric sort direction")
    if type(data.get("clear_steps", False)) is not bool:
        raise ValueError("The Clear Steps column flag must be true or false")
    stages = data["stages"]
    stage_ids = [stage["id"] for stage in stages]
    if not stage_ids or any(not stage_id for stage_id in stage_ids) or len(set(stage_ids)) != len(stage_ids):
        raise ValueError("Stages need unique IDs in their canonical order")
    if any(type(stage.get("solved")) not in (bool, type(None)) for stage in stages):
        raise ValueError("Solved must be true, false, or unavailable")
    seeds = {stage["evaluation_seeds"] for stage in stages if stage.get("evaluation_seeds") is not None}
    if any(type(count) is not int or count < 1 for count in seeds):
        raise ValueError("Evaluation seed counts must be positive integers")
    collection = data["collection_url"]
    if not re.fullmatch(r"https://huggingface\.co/collections/[^/]+/[^/?#]+", collection):
        raise ValueError("Expected a public Hugging Face collection URL")

    properties = [
        ("Provider", linked(data["provider"], data["provider_url"])),
        ("Environment", escape(data["environment"])),
        ("Framework", linked("GradLab", "https://github.com/tsilva/gradlab")),
        ("Eval Metric", escape(data["metric_name"])),
    ]
    if seeds:
        properties.append(("Eval Seeds", str(next(iter(seeds))) if len(seeds) == 1 else "Varies by stage"))
    show_solved = any(stage.get("solved") is not None for stage in stages)
    show_clear_steps = data.get("clear_steps", False)
    summary = ""
    headings = []
    if show_solved:
        solved = sum(stage.get("solved") is True for stage in stages)
        summary = f'<p class="summary"><strong>{solved} of {len(stages)} stages solved.</strong></p>'
        headings.append('<th class="solved" scope="col">Solved</th>')
    headings.extend((
        heading("Stage", "stage"),
        '<th scope="col">Algorithm</th>',
        heading(data["metric_column"], "metric", data["metric_direction"]),
        heading("Training steps", "trainingSteps"),
    ))
    if show_clear_steps:
        headings.append(heading("Clear Steps", "clearSteps"))
    headings.append('<th scope="col">Links</th>')
    return Template(ASSET.read_text(encoding="utf-8")).substitute(
        title=escape(f'{data["environment"]} Dashboard'),
        collection_url=escape(collection),
        summary=summary,
        properties="\n    ".join(f"<div><dt>{name}</dt><dd>{value}</dd></div>" for name, value in properties),
        methods=escape(data["methods"]),
        compact_columns=len(headings) - 1,
        headings="\n        ".join(headings),
        stage_rows="\n      ".join(render_row(stage, show_solved, show_clear_steps) for stage in stages),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    html = render(data)
    manifest = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "index.html").write_text(html, encoding="utf-8")
    (args.output_directory / "dashboard.json").write_text(manifest, encoding="utf-8")


if __name__ == "__main__":
    main()
