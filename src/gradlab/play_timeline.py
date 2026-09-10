"""Bounded episode-wide event overview, independent of inspection windows."""

from copy import deepcopy


class EventOverview:
    def __init__(self, episode_id: str, limit: int = 4096):
        self.episode_id = episode_id
        self.limit = limit
        self.width = 1
        self.through_step = -1
        self.buckets = {}

    def _merge(self, point):
        key = point["step"] // self.width
        previous = self.buckets.get(key)
        if previous is None:
            self.buckets[key] = deepcopy(point)
            return
        previous["last_step"] = max(previous["last_step"], point["last_step"])
        previous["count"] += point["count"]
        previous["boundary"] |= point["boundary"]
        previous["events"] = sorted(set(previous["events"]) | set(point["events"]))[:32]

    def append(self, point):
        step = point["step"]
        if step <= self.through_step:
            return
        self.through_step = step
        if not point.get("events") and not point.get("boundary"):
            return
        self._merge(
            {
                "step": step,
                "last_step": step,
                "count": 1,
                "boundary": bool(point.get("boundary")),
                "events": sorted(set(point.get("events") or []))[:32],
            }
        )
        while len(self.buckets) > self.limit:
            points = list(self.buckets.values())
            self.width *= 2
            self.buckets = {}
            for item in points:
                self._merge(item)

    def payload(self):
        return {
            "episode_id": self.episode_id,
            "bucket_size": self.width,
            "through_step": self.through_step,
            "points": deepcopy(list(self.buckets.values())),
        }
