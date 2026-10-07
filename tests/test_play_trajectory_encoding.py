"""Recording fast paths preserve data-only encoding and metadata redaction."""

from dataclasses import dataclass

import numpy as np

from gradlab.play_trajectory import (
    decode_tree,
    encode_tree,
    pack_record,
    portable_metadata,
    unpack_record,
)


def test_recording_codec_preserves_scalar_types_and_owned_array_bytes():
    image = np.arange(12, dtype=">i2").reshape(3, 4)[:, ::2]
    value = {
        "scalars": [None, True, "array", 2**70, 1.25],
        "tuple": ("scalar", "dict"),
        "numpy_scalar": np.float64(1.25),
        "nonfinite": float("nan"),
        "image": image,
    }
    tree = encode_tree(value)
    packed = pack_record(value)
    image[:] = -1
    for restored in (decode_tree(tree), unpack_record(packed)):
        assert restored["scalars"] == value["scalars"]
        assert type(restored["scalars"][1]) is bool
        assert type(restored["scalars"][3]) is int
        assert restored["tuple"] == ("scalar", "dict")
        assert restored["numpy_scalar"].shape == ()
        assert restored["numpy_scalar"].dtype == np.dtype("float64")
        assert np.isnan(restored["nonfinite"])
        assert restored["image"].dtype == np.dtype(">i2")
        np.testing.assert_array_equal(restored["image"], [[0, 2], [4, 6], [8, 10]])


def test_repeated_metadata_keys_do_not_cache_values_or_retain_private_fields(tmp_path):
    @dataclass
    class Facts:
        score: int
        api_token: str

    first = {
        "facts": Facts(3, "private"),
        "host": "private",
        "API_KEY": "private",
        "nested": [{"endpoint": "private", "reward": np.float32(0.5)}],
        "location": tmp_path,
        "source": "https://example.com/model?token=private",
    }
    sanitized = portable_metadata(first)
    first["facts"].score = 9
    first["source"] = "public-reference"
    later = portable_metadata(first)
    assert sanitized["facts"] == {"score": 3}
    assert later["facts"] == {"score": 9}
    assert sanitized["source"] == "[private location omitted]"
    assert later["source"] == "public-reference"
    assert sanitized["location"] == "[local path omitted]"
    assert "host" not in sanitized and "API_KEY" not in sanitized
    assert set(sanitized["nested"][0]) == {"reward"}
    assert type(sanitized["nested"][0]["reward"]) is np.float32
