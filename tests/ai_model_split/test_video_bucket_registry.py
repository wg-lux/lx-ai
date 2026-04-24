from __future__ import annotations

import json
from pathlib import Path

import pytest

from lx_ai.ai_model_split.video_bucket_registry import VideoBucketRegistry


def test_video_bucket_registry_new_file_starts_empty(tmp_path: Path) -> None:
    # checks missing registry file creates empty registry
    path = tmp_path / "video_bucket_registry.json"

    reg = VideoBucketRegistry.load(path=path, num_buckets=5)

    assert reg.path == path
    assert reg.num_buckets == 5
    assert reg.version == 1
    assert reg.videos == {}


def test_video_bucket_registry_set_and_get_value(tmp_path: Path) -> None:
    # checks video bucket can be set and read back
    path = tmp_path / "video_bucket_registry.json"
    reg = VideoBucketRegistry.load(path=path, num_buckets=5)

    reg.set("video:100", 3)

    assert reg.get("video:100") == 3


def test_video_bucket_registry_get_missing_value_returns_none(tmp_path: Path) -> None:
    # checks missing video key returns None
    path = tmp_path / "video_bucket_registry.json"
    reg = VideoBucketRegistry.load(path=path, num_buckets=5)

    assert reg.get("video:missing") is None


def test_video_bucket_registry_rejects_negative_bucket_id(tmp_path: Path) -> None:
    # checks bucket id cannot be negative
    path = tmp_path / "video_bucket_registry.json"
    reg = VideoBucketRegistry.load(path=path, num_buckets=5)

    with pytest.raises(ValueError, match="bucket_id out of range"):
        reg.set("video:100", -1)


def test_video_bucket_registry_rejects_bucket_id_equal_to_num_buckets(tmp_path: Path) -> None:
    # checks bucket id must be smaller than num_buckets
    path = tmp_path / "video_bucket_registry.json"
    reg = VideoBucketRegistry.load(path=path, num_buckets=5)

    with pytest.raises(ValueError, match="bucket_id out of range"):
        reg.set("video:100", 5)


def test_video_bucket_registry_to_dict_returns_sorted_plain_dict(tmp_path: Path) -> None:
    # checks to_dict returns stable sorted json safe structure
    path = tmp_path / "video_bucket_registry.json"
    reg = VideoBucketRegistry.load(path=path, num_buckets=5)

    reg.set("video:200", 2)
    reg.set("video:100", 1)

    data = reg.to_dict()

    assert data == {
        "version": 1,
        "num_buckets": 5,
        "videos": {
            "video:100": 1,
            "video:200": 2,
        },
    }


def test_video_bucket_registry_save_writes_file(tmp_path: Path) -> None:
    # checks save creates registry file on disk
    path = tmp_path / "nested" / "video_bucket_registry.json"
    reg = VideoBucketRegistry.load(path=path, num_buckets=5)

    reg.set("video:100", 1)
    reg.save()

    assert path.exists()


def test_video_bucket_registry_save_writes_expected_json(tmp_path: Path) -> None:
    # checks saved json content is correct
    path = tmp_path / "video_bucket_registry.json"
    reg = VideoBucketRegistry.load(path=path, num_buckets=5)

    reg.set("video:100", 1)
    reg.set("video:200", 2)
    reg.save()

    raw = json.loads(path.read_text(encoding="utf-8"))

    assert raw == {
        "version": 1,
        "num_buckets": 5,
        "videos": {
            "video:100": 1,
            "video:200": 2,
        },
    }


def test_video_bucket_registry_saved_file_can_be_loaded_again(tmp_path: Path) -> None:
    # checks saved registry can be loaded in next run
    path = tmp_path / "video_bucket_registry.json"

    reg = VideoBucketRegistry.load(path=path, num_buckets=5)
    reg.set("video:100", 1)
    reg.set("video:200", 2)
    reg.save()

    loaded = VideoBucketRegistry.load(path=path, num_buckets=5)

    assert loaded.get("video:100") == 1
    assert loaded.get("video:200") == 2
    assert loaded.num_buckets == 5
    assert loaded.version == 1


def test_video_bucket_registry_refuses_num_bucket_mismatch(tmp_path: Path) -> None:
    # checks registry refuses loading if num_buckets changed
    path = tmp_path / "video_bucket_registry.json"

    reg = VideoBucketRegistry.load(path=path, num_buckets=5)
    reg.set("video:100", 1)
    reg.save()

    with pytest.raises(ValueError, match="was created with num_buckets=5"):
        VideoBucketRegistry.load(path=path, num_buckets=6)


def test_video_bucket_registry_loads_missing_version_as_one(tmp_path: Path) -> None:
    # checks old registry files without version still load as version 1
    path = tmp_path / "video_bucket_registry.json"
    path.write_text(
        json.dumps(
            {
                "num_buckets": 5,
                "videos": {
                    "video:100": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    reg = VideoBucketRegistry.load(path=path, num_buckets=5)

    assert reg.version == 1
    assert reg.get("video:100") == 1


def test_video_bucket_registry_load_casts_video_keys_and_bucket_values(tmp_path: Path) -> None:
    # checks loaded video keys become strings and bucket ids become ints
    path = tmp_path / "video_bucket_registry.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "num_buckets": 5,
                "videos": {
                    "100": "2",
                },
            }
        ),
        encoding="utf-8",
    )

    reg = VideoBucketRegistry.load(path=path, num_buckets=5)

    assert reg.get("100") == 2


def test_video_bucket_registry_save_uses_tmp_file_then_replace(tmp_path: Path) -> None:
    # checks temporary file is not left after successful save
    path = tmp_path / "video_bucket_registry.json"
    tmp_file = path.with_suffix(path.suffix + ".tmp")

    reg = VideoBucketRegistry.load(path=path, num_buckets=5)
    reg.set("video:100", 1)
    reg.save()

    assert path.exists()
    assert not tmp_file.exists()


def test_video_bucket_registry_update_existing_video_bucket(tmp_path: Path) -> None:
    # checks setting same video key again updates bucket value
    path = tmp_path / "video_bucket_registry.json"
    reg = VideoBucketRegistry.load(path=path, num_buckets=5)

    reg.set("video:100", 1)
    reg.set("video:100", 3)

    assert reg.get("video:100") == 3


def test_video_bucket_registry_empty_registry_save_and_reload(tmp_path: Path) -> None:
    # checks empty registry can be saved and loaded
    path = tmp_path / "video_bucket_registry.json"

    reg = VideoBucketRegistry.load(path=path, num_buckets=5)
    reg.save()

    loaded = VideoBucketRegistry.load(path=path, num_buckets=5)

    assert loaded.videos == {}
    assert loaded.num_buckets == 5