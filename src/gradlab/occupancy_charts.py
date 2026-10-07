"""W&B historical inspection of bounded occupancy windows."""

from gradlab.json_utils import canonical_json_sha256
from gradlab.occupancy import OCCUPANCY_TABLE, ROW_FIELDS, WINDOW_FIELDS

CHART_NAME = "gradlab-occupancy-v1"


def chart_spec():
    def field(name, kind="quantitative", **kwargs):
        return {"field": "${field:" + name + "}", "type": kind, **kwargs}

    tooltips = [
        field(
            name,
            "nominal"
            if name
            in {
                "label",
                "origin",
                "segment",
                "run_id",
                "attempt_id",
                "cell_space_hash",
                "complete",
                "uncovered_interval",
            }
            else "quantitative",
        )
        for name in (
            "label",
            "origin",
            "count",
            "entries",
            "denominator",
            "fraction",
            "cumulative_count",
            "cumulative_denominator",
            "start_step",
            "end_step",
            "complete",
            "segment",
            "uncovered_interval",
            "cell_space_hash",
            "run_id",
            "attempt_id",
        )
    ]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"name": "wandb"},
        "transform": [
            {
                "window": [{"op": "row_number", "as": "delivery_row"}],
                "groupby": [
                    "${field:run_id}",
                    "${field:segment}",
                    "${field:sequence}",
                    "${field:cell}",
                    "${field:origin}",
                ],
            },
            {"filter": "datum.delivery_row === 1"},
            {
                "calculate": "'${string:grouping}' === 'full' ? datum.label : '${string:grouping}' === 'first dimension' ? '[' + datum['${field:dimension_0}'] + ']' : '[' + datum['${field:dimension_0}'] + ', ' + datum['${field:dimension_1}'] + ']'",
                "as": "view_label",
            },
            {
                "aggregate": [
                    *[
                        {"op": "sum", "field": "${field:" + name + "}", "as": name}
                        for name in ("count", "entries", "cumulative_count", "cumulative_entries")
                    ],
                    *[
                        {"op": "max", "field": "${field:" + name + "}", "as": name}
                        for name in ("denominator", "cumulative_denominator")
                    ],
                ],
                "groupby": [
                    "view_label",
                    "${field:origin}",
                    *["${field:" + name + "}" for name in WINDOW_FIELDS],
                ],
            },
            {
                "calculate": "datum.denominator > 0 ? datum.count / datum.denominator : null",
                "as": "fraction",
            },
            {"calculate": "datum.view_label", "as": "label"},
            {
                "calculate": "datum.run_id + ' · ' + substring(datum.segment, 0, 8) + ' · ' + substring(datum.cell_space_hash, 0, 8)",
                "as": "series",
            },
        ],
        "vconcat": [
            {
                "title": "${string:title}",
                "selection": {
                    "window": {
                        "type": "single",
                        "fields": ["${field:end_step}", "${field:run_id}", "${field:segment}"],
                        "empty": "none",
                    },
                },
                "transform": [{"filter": "datum.origin === '${string:origin}'"}],
                "mark": "rect",
                "width": 500,
                "height": 200,
                "encoding": {
                    "x": field(
                        "start_step", title="Collected policy transitions", scale={"zero": False}
                    ),
                    "x2": {"field": "${field:end_step}"},
                    "y": field("label", "nominal", title="Cell"),
                    "color": field(
                        "fraction", title="Occupancy fraction", scale={"domain": [0, 1]}
                    ),
                    "tooltip": tooltips,
                    "row": {
                        "field": "series",
                        "type": "nominal",
                        "title": "Run / segment / contract",
                        "header": {"labelLimit": 140},
                    },
                },
            },
            {
                "title": "Selected window. Empty origins have unavailable fractions.",
                "transform": [
                    {"filter": "datum.origin === '${string:origin}'"},
                    {"filter": {"selection": "window"}},
                ],
                "mark": "bar",
                "width": 500,
                "height": 160,
                "encoding": {
                    "x": field("label", "nominal", title="Cell"),
                    "y": field("fraction", title="Occupancy fraction", scale={"domain": [0, 1]}),
                    "tooltip": tooltips,
                    "color": field("run_id", "nominal", title="Run"),
                },
            },
        ],
    }


def chart_name():
    return f"{CHART_NAME}-{canonical_json_sha256(chart_spec())[:12]}"


def ensure_chart(api, *, entity):
    from wandb.errors import CommError

    identifier = f"{entity}/{chart_name()}"
    try:
        return api.create_custom_chart(
            entity=entity,
            name=chart_name(),
            display_name="Collected cell occupancy",
            spec_type="vega2",
            access="private",
            spec=chart_spec(),
        )
    except CommError as error:
        if "HTTP 409:" not in str(error):
            raise
        return identifier


def workspace_panel(wr, *, entity, layout, recent=False):
    return wr.CustomChart(
        query={
            "id": {},
            "name": {},
            **(
                {"summaryTable": {"tableKey": OCCUPANCY_TABLE}}
                if recent
                else {"historyTable": {"extraKeys": [], "tableKey": OCCUPANCY_TABLE, "index": 0}}
            ),
        },
        chart_name=f"{entity}/{chart_name()}",
        chart_strings={
            "origin": "combined",
            "grouping": "full",
            "title": (
                "Recent occupancy page — select a window"
                if recent
                else "Historical occupancy — Edit panel: Query index selects an earlier table"
            ),
        },
        chart_fields={
            name: name for name in (*WINDOW_FIELDS, *ROW_FIELDS, "dimension_0", "dimension_1")
        },
        layout=layout,
    )


def cumulative_chart_spec():
    def field(name, kind="quantitative", **kwargs):
        return {"field": "${field:" + name + "}", "type": kind, **kwargs}

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"name": "wandb"},
        "transform": [
            {"filter": "datum['${field:origin}'] === '${string:origin}'"},
            {
                "joinaggregate": [{"op": "max", "field": "${field:end_step}", "as": "latest_end"}],
                "groupby": ["${field:run_id}", "${field:segment}", "${field:cell_space_hash}"],
            },
            {"filter": "datum['${field:end_step}'] === datum.latest_end"},
            {
                "calculate": "datum['${field:cumulative_denominator}'] > 0 ? datum['${field:cumulative_count}'] / datum['${field:cumulative_denominator}'] : null",
                "as": "share",
            },
            {
                "calculate": "datum['${field:run_id}'] + ' · ' + substring(datum['${field:segment}'], 0, 8) + ' · ' + substring(datum['${field:cell_space_hash}'], 0, 8)",
                "as": "series",
            },
        ],
        "title": "Cumulative experience by cell — latest collection segment",
        "mark": "bar",
        "width": 650,
        "height": 220,
        "encoding": {
            "y": field("label", "nominal", title="Cell"),
            "x": {
                "field": "share",
                "type": "quantitative",
                "title": "Share of collected policy transitions",
                "scale": {"domain": [0, 1]},
                "axis": {"format": ".0%"},
            },
            "row": {
                "field": "series",
                "type": "nominal",
                "title": "Run / segment / contract",
                "header": {"labelLimit": 140},
            },
            "tooltip": [
                field("label", "nominal"),
                {"field": "share", "type": "quantitative", "format": ".3%"},
                field("cumulative_count", format=","),
                field("cumulative_denominator", format=","),
                field("end_step", format=","),
                field("origin", "nominal"),
                field("run_id", "nominal"),
                field("segment", "nominal"),
                field("cell_space_hash", "nominal"),
            ],
        },
    }


def cumulative_chart_name():
    return f"gradlab-occupancy-cumulative-v1-{canonical_json_sha256(cumulative_chart_spec())[:12]}"


def ensure_cumulative_chart(api, *, entity):
    from wandb.errors import CommError

    try:
        return api.create_custom_chart(
            entity=entity,
            name=cumulative_chart_name(),
            display_name="Cumulative cell occupancy",
            spec_type="vega2",
            access="private",
            spec=cumulative_chart_spec(),
        )
    except CommError as error:
        if "HTTP 409:" not in str(error):
            raise
        return f"{entity}/{cumulative_chart_name()}"


def cumulative_workspace_panel(wr, *, entity, layout):
    return wr.CustomChart(
        query={"id": {}, "name": {}, "summaryTable": {"tableKey": OCCUPANCY_TABLE}},
        chart_name=f"{entity}/{cumulative_chart_name()}",
        chart_strings={"origin": "combined"},
        chart_fields={name: name for name in (*WINDOW_FIELDS, *ROW_FIELDS)},
        layout=layout,
    )


def curriculum_chart_spec():
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"name": "wandb"},
        "hconcat": [
            {
                "title": title,
                "mark": "bar",
                "width": 240,
                "height": 300,
                "encoding": {
                    "y": {"field": "${field:cell}", "type": "nominal", "title": "Restorable cell"},
                    "row": {
                        "field": "${field:run_id}",
                        "type": "nominal",
                        "title": "Run",
                        "header": {"labelLimit": 140},
                    },
                    "x": {
                        "field": "${field:" + measure + "}",
                        "type": "quantitative",
                        "title": title,
                    },
                    "tooltip": [
                        {"field": "${field:" + name + "}", "type": kind}
                        for name, kind in [
                            ("run_id", "nominal"),
                            ("cell", "nominal"),
                            ("representatives", "quantitative"),
                            ("probability", "quantitative"),
                            ("cold", "nominal"),
                            ("strategy", "nominal"),
                            ("rollout", "quantitative"),
                            ("archive_lanes", "quantitative"),
                            ("normal_lanes", "quantitative"),
                            ("recent_combined_count", "quantitative"),
                        ]
                    ],
                },
            }
            for title, measure in [
                ("Retained representatives", "representatives"),
                ("Intended start probability", "probability"),
            ]
        ],
    }


def curriculum_chart_name():
    return "gradlab-curriculum-v1-" + canonical_json_sha256(curriculum_chart_spec())[:12]


def ensure_curriculum_chart(api, *, entity):
    from wandb.errors import CommError

    identifier = f"{entity}/{curriculum_chart_name()}"
    try:
        return api.create_custom_chart(
            entity=entity,
            name=curriculum_chart_name(),
            display_name="Curriculum representatives and starts",
            spec_type="vega2",
            access="private",
            spec=curriculum_chart_spec(),
        )
    except CommError as error:
        if "HTTP 409:" not in str(error):
            raise
        return identifier


def curriculum_workspace_panel(wr, *, entity, layout):
    from gradlab.curriculum_reporting import TABLE, COLUMNS

    return wr.CustomChart(
        query={"id": {}, "name": {}, "summaryTable": {"tableKey": TABLE}},
        chart_name=f"{entity}/{curriculum_chart_name()}",
        chart_fields={name: name for name in (*COLUMNS, "run_id")},
        layout=layout,
    )
