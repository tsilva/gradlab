"""Data-only, immutable policy initialization and isolated critic fitting."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any
from urllib.error import URLError


def normalize_initial_policy(value: Any, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    fields = {"repo_id", "revision", "sha256", "size_bytes", "reset_value_head"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} must define exactly {sorted(fields)}")
    repo_id = value["repo_id"]
    if (
        not isinstance(repo_id, str)
        or re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*", repo_id
        ) is None
    ):
        raise ValueError(f"{label}.repo_id must identify a Hugging Face owner and model")
    revision = value["revision"]
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"{label}.revision must pin a full Hugging Face commit")
    if (
        not isinstance(value["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
    ):
        raise ValueError(f"{label}.sha256 must be a lowercase SHA-256")
    size = value["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= 1024**3:
        raise ValueError(f"{label}.size_bytes must be a positive integer at most 1 GiB")
    if not isinstance(value["reset_value_head"], bool):
        raise ValueError(f"{label}.reset_value_head must be boolean")
    return dict(value)


def initialize_policy(model: Any, spec: Mapping[str, Any], *, root: Path) -> None:
    import torch

    from gradlab.model_sources import _download_public_file

    url = f"https://huggingface.co/{spec['repo_id']}/resolve/{spec['revision']}/policy.pt"
    try:
        path = _download_public_file(
            url,
            root / spec["sha256"] / "policy.pt",
            expected_sha256=spec["sha256"],
            expected_size=spec["size_bytes"],
        )
    except URLError:
        raise RuntimeError("Initial policy download failed; provider details withheld") from None
    weights = torch.load(path, map_location="cpu", weights_only=True)
    expected = model.policy.state_dict()
    if not isinstance(weights, Mapping) or set(weights) != set(expected):
        raise ValueError("initial policy tensor names do not match the requested policy")
    for name, target in expected.items():
        value = weights[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != target.shape
            or value.dtype != target.dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"initial policy tensor is incompatible or nonfinite: {name}")
    model.policy.load_state_dict(weights, strict=True)
    if spec["reset_value_head"]:
        with torch.no_grad():
            for parameter in model.policy.value_net.parameters():
                parameter.zero_()
    model.policy.optimizer.state.clear()


def configure_critic_only(model: Any, *, enabled: bool) -> None:
    model.gradlab_critic_only = enabled
    if not enabled:
        for parameter in model.policy.parameters():
            parameter.requires_grad_(True)
        return
    policy = model.policy
    if policy.share_features_extractor:
        raise ValueError("critic_only requires separate actor and critic feature extractors")
    actor = {
        id(p)
        for module in (
            policy.pi_features_extractor,
            policy.mlp_extractor.policy_net,
            policy.action_net,
        )
        for p in module.parameters()
    }
    critic = {
        id(p)
        for module in (
            policy.vf_features_extractor,
            policy.mlp_extractor.value_net,
            policy.value_net,
        )
        for p in module.parameters()
    }
    if actor & critic:
        raise ValueError("critic_only requires separate actor and critic feature extractors")
    for parameter in policy.parameters():
        parameter.requires_grad_(id(parameter) in critic)


def preserve_actor_eval_mode(model: Any) -> None:
    if getattr(model, "gradlab_critic_only", False):
        for module in (
            model.policy.pi_features_extractor,
            model.policy.mlp_extractor.policy_net,
            model.policy.action_net,
        ):
            module.eval()
