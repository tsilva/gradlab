"""Live train/eval overlays with independent scientific step axes."""

from functools import lru_cache

from gradlab.json_utils import canonical_json_sha256


@lru_cache(maxsize=1)
def query_preserving_chart_class():
    from wandb_workspaces.reports.v2 import CustomChart

    class QueryPreservingChart(CustomChart):
        # The SDK's dict representation silently drops repeated history fields.
        # W&B needs one history query per sparse series, otherwise it inner-joins
        # train and eval rows and returns no data.
        def _to_model(self):
            model = super()._to_model()
            if hasattr(self, "_raw_query"):
                model.config.user_query = self._raw_query.model_copy(deep=True)
            return model

        @classmethod
        def _from_model(cls, model):
            obj = super()._from_model(model)
            object.__setattr__(obj, "_raw_query", model.config.user_query.model_copy(deep=True))
            return obj

    return QueryPreservingChart


def chart_spec():
    layers = []
    for source, label in (
        ("train", "Train (rolling 100 episodes)"),
        ("eval", "Eval (complete episode set)"),
    ):
        x, y = "${field:" + source + "_x}", "${field:" + source + "_y}"
        layers.append(
            {
                "transform": [
                    {"filter": f'isValid(datum["{x}"]) && isValid(datum["{y}"])'},
                    {"calculate": repr(label), "as": "series"},
                ],
                "mark": {"type": "line", "point": source == "eval"},
                "encoding": {
                    "x": {
                        "field": x,
                        "type": "quantitative",
                        "title": "Training timesteps",
                        "axis": {"format": "~s"},
                    },
                    "y": {
                        "field": y,
                        "type": "quantitative",
                        "title": "Mean normalized bricks destroyed",
                    },
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "title": None,
                        "scale": {
                            "domain": [
                                "Train (rolling 100 episodes)",
                                "Eval (complete episode set)",
                            ],
                            "range": ["#18a7bd", "#f6903d"],
                        },
                    },
                    "detail": {"field": "${field:run}", "type": "nominal"},
                    "tooltip": [
                        {"field": "${field:run}", "title": "Run"},
                        {"field": "series"},
                        {"field": x, "type": "quantitative", "title": "Timesteps", "format": ","},
                        {"field": y, "type": "quantitative", "title": "Mean", "format": ".4f"},
                    ],
                },
            }
        )
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Normalized bricks destroyed: train vs eval",
        "data": {"name": "wandb"},
        "layer": layers,
        "config": {"legend": {"orient": "top"}},
    }


def chart_name():
    return "gradlab-train-eval-bricks-v1-" + canonical_json_sha256(chart_spec())[:12]


def ensure_chart(api, *, entity):
    from wandb.errors import CommError

    try:
        return api.create_custom_chart(
            entity=entity,
            name=chart_name(),
            display_name="Normalized bricks destroyed: train vs eval",
            spec_type="vega2",
            access="private",
            spec=chart_spec(),
        )
    except CommError as error:
        if "HTTP 409:" not in str(error):
            raise
        return f"{entity}/{chart_name()}"


def workspace_panel(wr, panel, *, entity, layout):
    from wandb_workspaces.reports.v2 import internal

    fields = {"run": "name"}
    queries = [
        internal.QueryField(name="id", fields=[]),
        internal.QueryField(name="name", fields=[]),
    ]
    for source, metric in zip(("train", "eval"), panel.y, strict=True):
        step = source + "/step"
        fields.update({source + "_x": step, source + "_y": metric})
        queries.append(
            internal.QueryField(
                name="history",
                args=[internal.QueryField(name="keys", value=[step, metric])],
                fields=[],
            )
        )
    chart = query_preserving_chart_class()(
        query={"id": {}, "name": {}},
        chart_name=f"{entity}/{chart_name()}",
        chart_fields=fields,
        layout=layout,
    )
    raw = chart._to_model().config.user_query
    raw.query_fields[0].fields = queries
    object.__setattr__(chart, "_raw_query", raw)
    return chart
