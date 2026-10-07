"""Exact episode action counts, frozen at each recorded Playback cursor."""

from copy import deepcopy


class EpisodeActionSummary:
    def __init__(self, contract):
        self.space = ((contract or {}).get("policy") or {}).get("space") or {}
        self.tuples = self.space.get("legal_tuples")
        self.count = (
            len(self.tuples)
            if isinstance(self.tuples, list)
            else self.space.get("n", 0)
            if self.space.get("type") == "discrete"
            else 0
        )
        # Imported contracts are external data; bound per-cursor diagnostic storage.
        if type(self.count) is not int or not 0 < self.count <= 65536:
            self.count = 0
        self.start = self.space.get("start", 0)
        if type(self.start) is not int:
            self.count = 0
        self.value = None

    def _offset(self, action):
        if isinstance(action, list):
            if len(action) == 1 and type(action[0]) is int:
                action = action[0]
            elif isinstance(self.tuples, list):
                try:
                    return self.tuples.index(action)
                except ValueError:
                    return None
        if type(action) is int:
            offset = action - (0 if isinstance(self.tuples, list) else self.start)
            if 0 <= offset < self.count:
                return offset
        return None

    def append(self, transition):
        episode, step, sequence = (transition[key] for key in ("episode", "step", "sequence"))
        if self.value is None or self.value["episode"] != episode:
            self.value = {
                "episode": episode,
                "step": 0,
                "sequence": None,
                "status": "available" if self.count else "unsupported",
                **{
                    name: {
                        "counts": [0] * self.count,
                        "population_count": 0,
                        "missing_count": 0,
                        "unmappable_count": 0,
                    }
                    for name in ("policy", "environment")
                },
            }
        value = self.value
        if sequence == value["sequence"]:
            return
        if step != value["step"] + 1:
            value["status"] = "partial-history"
        value.update(step=step, sequence=sequence)
        if value["status"] != "available":
            return
        source = transition.get("action_source")
        decision = transition.get("decision") or {}
        for name, action, missing in (
            ("policy", decision.get("selected_action"), source != "policy"),
            ("environment", transition.get("effective_action"), False),
        ):
            if name == "policy" and source == "human":
                continue
            totals = value[name]
            totals["population_count"] += 1
            if missing or action is None:
                totals["missing_count"] += 1
            else:
                offset = self._offset(action)
                if offset is None:
                    totals["unmappable_count"] += 1
                else:
                    totals["counts"][offset] += 1

    def payload(self, transition):
        if transition is None or self.value is None:
            return None
        if any(self.value[key] != transition[key] for key in ("episode", "step", "sequence")):
            return None
        return deepcopy(self.value)
