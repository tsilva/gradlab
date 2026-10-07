"""CPU child for a frozen evaluation attempt; no W&B or control credentials."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from gradlab.file_utils import atomic_write_json
from gradlab.modal_eval_worker import execute_attempt


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise ValueError("expected one request path")
    request = Path(arguments[0])
    result_path = request.with_name("result.json")
    try:
        payload = json.loads(request.read_text(encoding="utf-8"))
        result = execute_attempt(payload, cache_root=request.parent / "asset-cache")
    except Exception as exc:
        atomic_write_json(result_path, {"status": "failed", "error": repr(exc)[:4000]})
        return 1
    atomic_write_json(result_path, {"status": "succeeded", "result": result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
