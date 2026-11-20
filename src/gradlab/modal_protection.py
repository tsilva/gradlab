"""Read the control-storage inventory that protects nonterminal evaluators."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor


def control_modal_app_names() -> tuple[str, ...]:
    from gradlab.r2_store import BucketConfig, R2Bucket

    control = R2Bucket(BucketConfig.from_env("GRADLAB_CONTROL_R2"))
    keys = [key for key in control.iter_keys("runs/") if key.startswith("runs/")]
    manifests = {
        key.split("/")[1]: key
        for key in keys
        if key.count("/") == 2 and key.endswith("/manifest.json")
    }
    terminal_run_ids = {
        key.split("/")[1] for key in keys if key.count("/") == 2 and key.endswith("/terminal.json")
    }
    outstanding = [key for run_id, key in manifests.items() if run_id not in terminal_run_ids]
    with ThreadPoolExecutor(max_workers=16) as pool:
        protected = list(pool.map(control.get_json, outstanding))
    return tuple(sorted({
        app_name
        for manifest in protected
        if (app_name := str((manifest.get("modal") or {}).get("app_name") or "").strip())
    }))


def main() -> int:
    from gradlab.training_vault import validate_session

    try:
        validate_session()
        names = control_modal_app_names()
    except Exception:
        print("Modal protection inventory unavailable", file=sys.stderr)
        return 1
    print(json.dumps({"complete": True, "protected_modal_apps": names}))
    return 0


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("Modal protection inventory does not accept arguments")
    raise SystemExit(main())
