from __future__ import annotations


def require_cuda() -> None:
    """Reject unavailable CUDA before a trainer can fall back to CPU."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this training run but is unavailable; "
            "verify the NVIDIA driver and container GPU assignment. "
            "CPU fallback is not permitted."
        )


def resolve_sb3_device(device: str) -> str:
    """Resolve SB3's `auto` device with Apple MPS support."""
    if device.partition(":")[0] == "cuda":
        require_cuda()
        return device
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
