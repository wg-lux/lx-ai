# lx_ai/training/bucket_snapshot.py
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def create_run_folder() -> Path:
    """
    Creates new timestamped run folder.
    """
    base_bucket_path = Path(
        os.getenv("BUCKET_SNAPSHOT_DIR", "data/model_training/buckets")
    ).expanduser()

    base_bucket_path.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_path = base_bucket_path / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    return run_path


def save_bucket_snapshot(
    *,
    bucket_map: dict[str, int],
    train_buckets: list[int],
    val_buckets: list[int],
    test_buckets: list[int],
    dataset_ids: list[int],
    bucket_policy: dict[str, Any],
) -> None:
    """
    Save full bucket snapshot to disk.
    """
    run_path = create_run_folder()

    with open(run_path / "all_bucket_assignments.json", "w", encoding="utf-8") as f:
        json.dump(bucket_map, f, indent=2)

    with open(run_path / "training_buckets.json", "w", encoding="utf-8") as f:
        json.dump(train_buckets, f, indent=2)

    with open(run_path / "validation_buckets.json", "w", encoding="utf-8") as f:
        json.dump(val_buckets, f, indent=2)

    with open(run_path / "test_buckets.json", "w", encoding="utf-8") as f:
        json.dump(test_buckets, f, indent=2)

    with open(run_path / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_ids": dataset_ids,
                "bucket_policy": bucket_policy,
            },
            f,
            indent=2,
        )

    print(f"Bucket snapshot saved to {run_path}")
