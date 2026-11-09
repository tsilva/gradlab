from __future__ import annotations

from unittest.mock import patch

import pytest

from gradlab.device import resolve_sb3_device


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_explicit_cuda_rejects_cpu_fallback(device: str) -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="CPU fallback is not permitted"),
    ):
        resolve_sb3_device(device)


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_explicit_cuda_keeps_requested_device_when_available(device: str) -> None:
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_sb3_device(device) == device


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_remains_portable_outside_gpu_allocations(
    cuda: bool, mps: bool, expected: str,
) -> None:
    with (
        patch("torch.cuda.is_available", return_value=cuda),
        patch("torch.backends.mps.is_available", return_value=mps),
    ):
        assert resolve_sb3_device("auto") == expected


def test_explicit_cpu_does_not_probe_cuda() -> None:
    with patch("torch.cuda.is_available") as available:
        assert resolve_sb3_device("cpu") == "cpu"
    available.assert_not_called()
