from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import random

import numpy as np
import torch


class SmallCnn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        arr = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(arr.shape).encode())
        digest.update(str(arr.dtype).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def worker(idx: int, args: argparse.Namespace, queue: mp.Queue) -> None:
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    model = SmallCnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, eps=1e-8)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1000)
    x = torch.rand((args.batch_size, 4, 84, 84), generator=generator, device=device)
    y = torch.randint(0, 7, (args.batch_size,), generator=generator, device=device)

    initial = state_hash(model)
    for _ in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    queue.put((idx, initial, state_hash(model), float(loss.detach().cpu())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(idx, args, queue)) for idx in range(2)]
    for proc in procs:
        proc.start()
    results = [queue.get() for _ in procs]
    for proc in procs:
        proc.join()
    for result in sorted(results):
        print(result)
    finals = {result[2] for result in results}
    print("final_hashes_equal", len(finals) == 1)


if __name__ == "__main__":
    main()
