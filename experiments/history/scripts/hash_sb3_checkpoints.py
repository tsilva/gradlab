from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from stable_baselines3 import PPO


def policy_hash(path: Path) -> str:
    model = PPO.load(path, device="cpu")
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        arr = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(arr.shape).encode())
        digest.update(str(arr.dtype).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_glob")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    for run_dir in sorted(Path("runs").glob(args.runs_glob)):
        print("RUN", run_dir.name)
        checkpoints = sorted((run_dir / "checkpoints").glob("ppo_*_steps.zip"))
        print("checkpoint_count", len(checkpoints))
        for checkpoint in checkpoints[: args.limit]:
            print(checkpoint.name, policy_hash(checkpoint))


if __name__ == "__main__":
    main()
