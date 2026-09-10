"""Constant-memory reward sufficient statistics, frozen at each recorded cursor."""

from copy import deepcopy
import math


class EpisodeRewardSummary:
    def __init__(self):
        self.value = None

    def append(self, transition, contract):
        episode, step, sequence = (transition[key] for key in ("episode", "step", "sequence"))
        if self.value is None or self.value["episode"] != episode:
            self.value = {
                "episode": episode,
                "step": 0,
                "sequence": None,
                "status": "available",
                "raw": 0.0,
                "preclip": 0.0,
                "final": 0.0,
                "entries": {},
            }
        value = self.value
        if value["sequence"] == sequence:
            return
        if step != value["step"] + 1:
            value.update(
                status="partial-history",
                message="Episode accounting requires a complete recording from step 1.",
            )
        value.update(step=step, sequence=sequence)
        if value["status"] != "available":
            return
        reward = transition["reward"]
        try:
            if contract.get("status") != "available":
                raise ValueError(
                    contract.get("reason") or "Reward transform contract is unavailable."
                )
            if reward.get("accounting_error"):
                raise ValueError(reward["accounting_error"])
            raw, final, scale = (
                self._number(x)
                for x in (reward.get("raw"), reward.get("shaped"), contract.get("reward_scale"))
            )
            if not 0 <= scale <= 1:
                raise ValueError("Reward scale is invalid.")
            preclip = raw * scale
            clip = contract.get("clip_bounds")
            expected = preclip
            if clip is not None:
                if not isinstance(clip, list) or len(clip) != 2:
                    raise ValueError("Reward clip bounds are invalid.")
                low, high = map(self._number, clip)
                if low > high:
                    raise ValueError("Reward clip bounds are invalid.")
                expected = min(high, max(low, preclip))
            if abs(final - expected) > max(1e-9, 1e-6 * (abs(raw) + abs(preclip) + abs(final))):
                raise ValueError("Final reward does not match scale-then-clip accounting.")
            components = reward.get("components")
            if not isinstance(components, dict):
                raise ValueError("Reward components are malformed.")
            if len(value["entries"].keys() | components.keys()) > 64:
                raise ValueError("Reward component inventory exceeds the accounting limit.")
            entries = {
                key: (self._number(number), self._number(number) * scale)
                for key, number in components.items()
            }
            residual = raw - sum(number for number, _ in entries.values())
            entries["unattributed"] = (residual, residual * scale)
            entries["clip_adjustment"] = (None, final - preclip)
            value["raw"] += raw
            value["preclip"] += preclip
            value["final"] += final
            for key, (number, impact) in entries.items():
                entry = value["entries"].setdefault(
                    key, {"raw": None if number is None else 0.0, "impact": 0.0, "magnitude": 0.0}
                )
                if number is not None:
                    entry["raw"] += number
                entry["impact"] += impact
                entry["magnitude"] += abs(impact)
            total = self._number(reward.get("return"))
            gross = sum(entry["magnitude"] for entry in value["entries"].values())
            if abs(value["final"] - total) > max(1e-9, 1e-6 * (gross + abs(total))):
                raise ValueError(
                    "Recorded rewards do not reconcile to the authoritative episode return."
                )
        except (ValueError, TypeError, OverflowError) as exc:
            value.update(status="protocol-error", message=f"Step {step}: {exc}")

    @staticmethod
    def _number(value):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Reward evidence is missing or non-finite.")
        return float(value)

    def payload(self, transition):
        if transition is None or self.value is None:
            return None
        if any(self.value[key] != transition[key] for key in ("episode", "step", "sequence")):
            return None
        return deepcopy(self.value)
