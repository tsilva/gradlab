from __future__ import annotations

from unittest.mock import Mock

import pytest
from botocore.exceptions import ResponseStreamingError

from gradlab.r2_store import BucketConfig, ObjectReadInterrupted, R2Bucket


def test_missing_bucket_uri_names_the_exact_operator_variable() -> None:
    with pytest.raises(ValueError, match="GRADLAB_CONTROL_R2_URI is not set"):
        BucketConfig.from_env("GRADLAB_CONTROL_R2", environment={})


def test_visibly_truncated_endpoint_is_rejected_before_boto() -> None:
    with pytest.raises(ValueError, match="visibly truncated"):
        BucketConfig.from_env(
            "GRADLAB_EVAL_R2",
            environment={
                "GRADLAB_EVAL_R2_URI": "s3://eval",
                "GRADLAB_EVAL_R2_ENDPOINT_URL": "https://account.r2.cloudf…",
                "GRADLAB_EVAL_R2_ACCESS_KEY_ID": "access",
                "GRADLAB_EVAL_R2_SECRET_ACCESS_KEY": "secret",
            },
        )


def test_incomplete_endpoint_is_rejected_before_boto() -> None:
    with pytest.raises(ValueError, match="complete http:// or https:// URL"):
        BucketConfig.from_env(
            "GRADLAB_MODELS_R2",
            environment={
                "GRADLAB_MODELS_R2_URI": "s3://models",
                "GRADLAB_MODELS_R2_ENDPOINT_URL": "not-a-url",
                "GRADLAB_MODELS_R2_ACCESS_KEY_ID": "access",
                "GRADLAB_MODELS_R2_SECRET_ACCESS_KEY": "secret",
                "GRADLAB_MODELS_R2_PUBLIC_BASE_URL": "https://models.example",
            },
            public=True,
        )


def test_file_public_base_is_valid_for_local_simulation() -> None:
    config = BucketConfig(
        uri="file:///tmp/gradlab-models",
        public_base_url="file:///tmp/gradlab-models",
    )

    config.validate(public=True)


def test_incomplete_public_base_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="PUBLIC_BASE_URL must be a complete"):
        BucketConfig.from_env(
            "GRADLAB_MODELS_R2",
            environment={
                "GRADLAB_MODELS_R2_URI": "s3://models",
                "GRADLAB_MODELS_R2_ENDPOINT_URL": "https://r2.example",
                "GRADLAB_MODELS_R2_ACCESS_KEY_ID": "access",
                "GRADLAB_MODELS_R2_SECRET_ACCESS_KEY": "secret",
                "GRADLAB_MODELS_R2_PUBLIC_BASE_URL": "truncated",
            },
            public=True,
        )


def test_interrupted_s3_body_closes_and_withholds_provider_error() -> None:
    bucket = R2Bucket(BucketConfig(
        uri="s3://models",
        endpoint_url="https://r2.example",
        access_key_id="fake",
        secret_access_key="fake",
    ))
    provider_error = ResponseStreamingError(error="synthetic provider detail")
    assert provider_error.response is None
    body = Mock()
    body.read.side_effect = provider_error
    bucket._client = Mock(get_object=Mock(return_value={"Body": body}))

    with pytest.raises(ObjectReadInterrupted, match="R2 object response was interrupted") as error:
        bucket.get_bytes("immutable.zip")

    assert "synthetic provider detail" not in str(error.value)
    body.close.assert_called_once_with()
