from __future__ import annotations

import pytest

from rlab.r2_store import BucketConfig


def test_missing_bucket_uri_names_the_exact_operator_variable() -> None:
    with pytest.raises(ValueError, match="RLAB_CONTROL_R2_URI is not set"):
        BucketConfig.from_env("RLAB_CONTROL_R2", environment={})


def test_visibly_truncated_endpoint_is_rejected_before_boto() -> None:
    with pytest.raises(ValueError, match="visibly truncated"):
        BucketConfig.from_env(
            "RLAB_EVAL_R2",
            environment={
                "RLAB_EVAL_R2_URI": "s3://eval",
                "RLAB_EVAL_R2_ENDPOINT_URL": "https://account.r2.cloudf…",
                "RLAB_EVAL_R2_ACCESS_KEY_ID": "access",
                "RLAB_EVAL_R2_SECRET_ACCESS_KEY": "secret",
            },
        )


def test_incomplete_endpoint_is_rejected_before_boto() -> None:
    with pytest.raises(ValueError, match="complete http:// or https:// URL"):
        BucketConfig.from_env(
            "RLAB_MODELS_R2",
            environment={
                "RLAB_MODELS_R2_URI": "s3://models",
                "RLAB_MODELS_R2_ENDPOINT_URL": "not-a-url",
                "RLAB_MODELS_R2_ACCESS_KEY_ID": "access",
                "RLAB_MODELS_R2_SECRET_ACCESS_KEY": "secret",
                "RLAB_MODELS_R2_PUBLIC_BASE_URL": "https://models.example",
            },
            public=True,
        )


def test_file_public_base_is_valid_for_local_simulation() -> None:
    config = BucketConfig(
        uri="file:///tmp/rlab-models",
        public_base_url="file:///tmp/rlab-models",
    )

    config.validate(public=True)


def test_incomplete_public_base_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="PUBLIC_BASE_URL must be a complete"):
        BucketConfig.from_env(
            "RLAB_MODELS_R2",
            environment={
                "RLAB_MODELS_R2_URI": "s3://models",
                "RLAB_MODELS_R2_ENDPOINT_URL": "https://r2.example",
                "RLAB_MODELS_R2_ACCESS_KEY_ID": "access",
                "RLAB_MODELS_R2_SECRET_ACCESS_KEY": "secret",
                "RLAB_MODELS_R2_PUBLIC_BASE_URL": "truncated",
            },
            public=True,
        )
