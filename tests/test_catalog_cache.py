from __future__ import annotations

import threading
from pathlib import Path

from gradlab.catalog_cache import CatalogEntryCache


def test_catalog_entry_cache_keeps_independent_slots(tmp_path: Path) -> None:
    cache = CatalogEntryCache(tmp_path / "catalog")

    with cache.slot_lock("runs", "alpha"):
        cache.write("runs", "alpha", {"items": [{"run_id": "alpha"}]})
    with cache.slot_lock("runs", "bravo"):
        cache.write("runs", "bravo", {"items": [{"run_id": "bravo"}]})

    assert cache.read("runs", "alpha") == {"items": [{"run_id": "alpha"}]}
    assert cache.read("runs", "bravo") == {"items": [{"run_id": "bravo"}]}


def test_catalog_entry_cache_prunes_least_recently_used_entry(
    tmp_path: Path,
) -> None:
    cache = CatalogEntryCache(tmp_path / "catalog", max_entries=2)
    cache.write("runs", "alpha", {"value": "alpha"})
    cache.write("runs", "bravo", {"value": "bravo"})

    assert cache.read("runs", "alpha") == {"value": "alpha"}
    cache.write("runs", "charlie", {"value": "charlie"})

    assert cache.read("runs", "alpha") == {"value": "alpha"}
    assert cache.read("runs", "bravo") is None
    assert cache.read("runs", "charlie") == {"value": "charlie"}


def test_catalog_entry_cache_does_not_prune_a_locked_refresh(
    tmp_path: Path,
) -> None:
    cache = CatalogEntryCache(tmp_path / "catalog", max_entries=1)
    cache.write("runs", "alpha", {"value": "old"})
    locked = threading.Event()
    release = threading.Event()

    def hold_refresh() -> None:
        with cache.slot_lock("runs", "alpha"):
            locked.set()
            release.wait(timeout=5)
            cache.write("runs", "alpha", {"value": "fresh"})

    worker = threading.Thread(target=hold_refresh)
    worker.start()
    assert locked.wait(timeout=5)
    cache.write("runs", "bravo", {"value": "bravo"})
    assert cache.read("runs", "alpha") == {"value": "old"}
    release.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert cache.read("runs", "alpha") == {"value": "fresh"}
