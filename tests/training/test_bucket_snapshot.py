from __future__ import annotations

import json
from pathlib import Path

from lx_ai.training.bucket_snapshot import create_run_folder, save_bucket_snapshot


def _set_test_bucket_snapshot_dir(tmp_path: Path, monkeypatch) -> Path:
    # keeps tests isolated from real project data directory
    bucket_dir = tmp_path / "data" / "model_training" / "buckets"
    monkeypatch.setenv("BUCKET_SNAPSHOT_DIR", str(bucket_dir))
    return bucket_dir


class TestBucketSnapshot:
    def test_create_run_folder_creates_directory(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # checks create_run_folder creates a timestamped directory in BUCKET_SNAPSHOT_DIR
        bucket_dir = _set_test_bucket_snapshot_dir(tmp_path, monkeypatch)

        run_path = create_run_folder()

        assert run_path.exists()
        assert run_path.is_dir()
        assert run_path.parent == bucket_dir

    def test_save_bucket_snapshot_creates_all_expected_files(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        # checks save_bucket_snapshot writes all expected json files
        bucket_dir = _set_test_bucket_snapshot_dir(tmp_path, monkeypatch)

        save_bucket_snapshot(
            bucket_map={"video:100": 1, "video:200": 2},
            train_buckets=[0, 1, 2],
            val_buckets=[3],
            test_buckets=[4],
            dataset_ids=[1, 2],
            bucket_policy={
                "num_buckets": 5,
                "validation_buckets": [3],
                "test_buckets": [4],
                "train_buckets": [0, 1, 2],
            },
        )

        run_dirs = list(bucket_dir.iterdir())
        assert len(run_dirs) == 1

        run_path = run_dirs[0]

        assert (run_path / "all_bucket_assignments.json").exists()
        assert (run_path / "training_buckets.json").exists()
        assert (run_path / "validation_buckets.json").exists()
        assert (run_path / "test_buckets.json").exists()
        assert (run_path / "run_metadata.json").exists()

    def test_save_bucket_snapshot_writes_bucket_assignments(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        # checks all bucket assignments are written correctly
        bucket_dir = _set_test_bucket_snapshot_dir(tmp_path, monkeypatch)

        bucket_map = {"video:100": 1, "video:200": 2}

        save_bucket_snapshot(
            bucket_map=bucket_map,
            train_buckets=[0, 1, 2],
            val_buckets=[3],
            test_buckets=[4],
            dataset_ids=[1, 2],
            bucket_policy={"num_buckets": 5},
        )

        run_path = next(bucket_dir.iterdir())

        data = json.loads(
            (run_path / "all_bucket_assignments.json").read_text(encoding="utf-8")
        )

        assert data == bucket_map

    def test_save_bucket_snapshot_writes_split_bucket_files(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        # checks train validation and test bucket files are written correctly
        bucket_dir = _set_test_bucket_snapshot_dir(tmp_path, monkeypatch)

        save_bucket_snapshot(
            bucket_map={"video:100": 1},
            train_buckets=[0, 1, 2],
            val_buckets=[3],
            test_buckets=[4],
            dataset_ids=[1],
            bucket_policy={"num_buckets": 5},
        )

        run_path = next(bucket_dir.iterdir())

        train = json.loads(
            (run_path / "training_buckets.json").read_text(encoding="utf-8")
        )
        val = json.loads(
            (run_path / "validation_buckets.json").read_text(encoding="utf-8")
        )
        test = json.loads((run_path / "test_buckets.json").read_text(encoding="utf-8"))

        assert train == [0, 1, 2]
        assert val == [3]
        assert test == [4]

    def test_save_bucket_snapshot_writes_metadata(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        # checks run metadata contains dataset ids and bucket policy
        bucket_dir = _set_test_bucket_snapshot_dir(tmp_path, monkeypatch)

        bucket_policy = {
            "num_buckets": 5,
            "validation_buckets": [3],
            "test_buckets": [4],
            "train_buckets": [0, 1, 2],
        }

        save_bucket_snapshot(
            bucket_map={"video:100": 1},
            train_buckets=[0, 1, 2],
            val_buckets=[3],
            test_buckets=[4],
            dataset_ids=[1, 2],
            bucket_policy=bucket_policy,
        )

        run_path = next(bucket_dir.iterdir())

        metadata = json.loads(
            (run_path / "run_metadata.json").read_text(encoding="utf-8")
        )

        assert metadata == {
            "dataset_ids": [1, 2],
            "bucket_policy": bucket_policy,
        }
