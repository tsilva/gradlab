from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any

from gradlab.cli_parser import ExactArgumentParser
from gradlab.contract_versions import TRAIN_CONFIG_CONTRACT_SCHEMA_VERSION
from gradlab.json_utils import canonical_json_sha256, json_safe
from gradlab.train_config import (
    TRAIN_CONFIG_FIELDS,
    TrainConfigField,
    validate_and_normalize_train_config,
)
from gradlab.training_backend import training_backend_contract_payload


RUNTIME_DESCRIPTOR_SCHEMA_VERSION = 7
_CLI_ONLY_FIELD_METADATA = frozenset({"flag", "help"})


def _runtime_field_payload(field: TrainConfigField) -> dict[str, Any]:
    return {
        key: value for key, value in asdict(field).items() if key not in _CLI_ONLY_FIELD_METADATA
    }


def train_config_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": TRAIN_CONFIG_CONTRACT_SCHEMA_VERSION,
        "fields": [json_safe(_runtime_field_payload(field)) for field in TRAIN_CONFIG_FIELDS],
        "training_backends": json_safe(training_backend_contract_payload()),
    }


def train_config_contract_sha256() -> str:
    return canonical_json_sha256(
        train_config_contract_payload(),
        ensure_ascii=True,
    )


def runtime_contract(*, runtime_image_ref: str | None = None) -> dict[str, Any]:
    runtime_build_source_sha = os.environ.get("GRADLAB_SOURCE_SHA", "").strip()
    return {
        "schema_version": TRAIN_CONFIG_CONTRACT_SCHEMA_VERSION,
        "runtime_build_source_sha": runtime_build_source_sha,
        "runtime_input_sha256": os.environ.get("GRADLAB_RUNTIME_INPUT_SHA256", "").strip(),
        "runtime_image_ref": str(
            runtime_image_ref or os.environ.get("GRADLAB_MODAL_EVAL_RUNTIME_IMAGE", "")
        ).strip(),
        "train_config_contract_sha256": train_config_contract_sha256(),
    }


def validate_config_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("train config stdin must contain a JSON object")
    normalized = validate_and_normalize_train_config(
        payload,
        label="runtime preflight train config",
        required_keys=("training_backend",),
    )
    receipt = runtime_contract()
    receipt.update(
        {
            "validated": True,
            "validated_field_count": len(normalized),
            "validated_fields_sha256": canonical_json_sha256(
                sorted(normalized),
                ensure_ascii=True,
            ),
        }
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = ExactArgumentParser(
        description="Report and validate the immutable gradlab train-runtime contract."
    )
    parser.add_argument(
        "--validate-config-stdin",
        action="store_true",
        help="Validate one materialized train-config JSON object from stdin.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = (
        validate_config_payload(json.load(sys.stdin))
        if args.validate_config_stdin
        else runtime_contract()
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
