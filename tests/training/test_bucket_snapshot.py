from __future__ import annotations

import json
from pathlib import Path

from lx_ai.training.bucket_snapshot import create_run_folder, save_bucket_snapshot


class TestBucketSnapshot:
    def test_create_run_folder_creates_directory(self, tmp_path: Path, monkeypatch) -> None:
        # checks create_run_folder creates a timestamped directory
        monkeypatch.chdir(tmp_path)

        run_path = create_run_folder()

        assert run_path.exists()
        assert run_path.is_dir()
        assert str(run_path).startswith("data/model_training/buckets")

    def test_save_bucket_snapshot_creates_all_expected_files(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        # checks save_bucket_snapshot writes all expected json files
        monkeypatch.chdir(tmp_path)

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

        run_dirs = list((tmp_path / "data/model_training/buckets").iterdir())
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
        monkeypatch.chdir(tmp_path)

        bucket_map = {"video:100": 1, "video:200": 2}

        save_bucket_snapshot(
            bucket_map=bucket_map,
            train_buckets=[0, 1, 2],
            val_buckets=[3],
            test_buckets=[4],
            dataset_ids=[1, 2],
            bucket_policy={"num_buckets": 5},
        )

        run_path = next((tmp_path / "data/model_training/buckets").iterdir())
        data = json.loads((run_path / "all_bucket_assignments.json").read_text())

        assert data == bucket_map

    def test_save_bucket_snapshot_writes_split_bucket_files(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        # checks train validation and test bucket files are written correctly
        monkeypatch.chdir(tmp_path)

        save_bucket_snapshot(
            bucket_map={"video:100": 1},
            train_buckets=[0, 1, 2],
            val_buckets=[3],
            test_buckets=[4],
            dataset_ids=[1],
            bucket_policy={"num_buckets": 5},
        )

        run_path = next((tmp_path / "data/model_training/buckets").iterdir())

        assert json.loads((run_path / "training_buckets.json").read_text()) == [0, 1, 2]
        assert json.loads((run_path / "validation_buckets.json").read_text()) == [3]
        assert json.loads((run_path / "test_buckets.json").read_text()) == [4]

    def test_save_bucket_snapshot_writes_metadata(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        # checks run metadata contains dataset ids and bucket policy
        monkeypatch.chdir(tmp_path)

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

        run_path = next((tmp_path / "data/model_training/buckets").iterdir())
        metadata = json.loads((run_path / "run_metadata.json").read_text())

        assert metadata["dataset_ids"] == [1, 2]
        assert metadata["bucket_policy"] == bucket_policy