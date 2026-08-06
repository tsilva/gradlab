from __future__ import annotations

from typing import Self

import torch as th
from torch import nn
from torch.distributions import Categorical

from stable_baselines3.common.distributions import Distribution

from gradlab.action_codecs import LegalTupleMultiDiscrete


class LegalTupleCategoricalDistribution(Distribution):
    """Categorical legal rows scored by the sum of their selected axis logits."""

    distribution: Categorical

    def __init__(self, action_space: LegalTupleMultiDiscrete):
        super().__init__()
        self.action_dims = [int(value) for value in action_space.nvec.reshape(-1)]
        self.axis_count = len(self.action_dims)
        legal = th.as_tensor(action_space.legal_tuples, dtype=th.long)
        offsets = th.as_tensor(
            [0, *list(th.cumsum(th.as_tensor(self.action_dims[:-1]), dim=0).tolist())],
            dtype=th.long,
        )
        self._logit_indices_cpu = legal + offsets
        multipliers = []
        product = 1
        for cardinality in self.action_dims:
            multipliers.append(product)
            product *= cardinality
        lookup = th.full((product,), -1, dtype=th.long)
        flat_legal = (legal * th.as_tensor(multipliers, dtype=th.long)).sum(dim=1)
        lookup[flat_legal] = th.arange(len(legal), dtype=th.long)
        self._legal_tuples_cpu = legal
        self._multipliers_cpu = th.as_tensor(multipliers, dtype=th.long)
        self._row_lookup_cpu = lookup
        self.action_logits: th.Tensor | None = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, sum(self.action_dims))

    def proba_distribution(self, action_logits: th.Tensor) -> Self:
        indices = self._logit_indices_cpu.to(action_logits.device)
        row_logits = action_logits[:, indices].sum(dim=-1)
        self.action_logits = action_logits
        self.distribution = Categorical(logits=row_logits)
        return self

    def _rows(self, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        values = actions.long().reshape(-1, self.axis_count)
        nvec = th.as_tensor(self.action_dims, dtype=th.long, device=values.device)
        in_bounds = ((values >= 0) & (values < nvec)).all(dim=1)
        multipliers = self._multipliers_cpu.to(values.device)
        flat = (values.clamp(min=0) * multipliers).sum(dim=1)
        lookup = self._row_lookup_cpu.to(values.device)
        flat_in_bounds = (flat >= 0) & (flat < lookup.numel())
        safe_flat = flat.clamp(0, lookup.numel() - 1)
        rows = lookup[safe_flat]
        valid = in_bounds & flat_in_bounds & (rows >= 0)
        if not th.compiler.is_compiling() and not bool(valid.all().item()):
            raise ValueError("action is outside the configured legal-tuple support")
        return rows.clamp_min(0), valid

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        rows, valid = self._rows(actions)
        result = self.distribution.log_prob(rows)
        return th.where(valid, result, th.full_like(result, -th.inf))

    def row_indices(self, actions: th.Tensor) -> th.Tensor:
        """Return the categorical support row for each exact legal tuple."""

        rows, _valid = self._rows(actions)
        return rows

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def _actions_for_rows(self, rows: th.Tensor) -> th.Tensor:
        legal = self._legal_tuples_cpu.to(rows.device)
        return legal[rows]

    def sample(self) -> th.Tensor:
        return self._actions_for_rows(self.distribution.sample())

    def mode(self) -> th.Tensor:
        return self._actions_for_rows(th.argmax(self.distribution.logits, dim=1))

    def actions_from_params(
        self,
        action_logits: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        return self.proba_distribution(action_logits).get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)


__all__ = ["LegalTupleCategoricalDistribution"]
