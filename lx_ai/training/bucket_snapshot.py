import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


BASE_BUCKET_PATH = Path("data/model_training/buckets")


def create_run_folder() -> Path:
    """
    Creates new timestamped run folder.
    """
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_path = BASE_BUCKET_PATH / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def save_bucket_snapshot(
    *,
    bucket_map: Dict[str, int],
    train_buckets: List[int],
    val_buckets: List[int],
    test_buckets: List[int],
    dataset_ids: List[int],
    bucket_policy: Dict[str, Any],
):
    """
    Save full bucket snapshot to disk.
    """

    run_path = create_run_folder()

    # Save all bucket assignments
    with open(run_path / "all_bucket_assignments.json", "w") as f:
        json.dump(bucket_map, f, indent=2)

    # Save split definitions
    with open(run_path / "training_buckets.json", "w") as f:
        json.dump(train_buckets, f, indent=2)

    with open(run_path / "validation_buckets.json", "w") as f:
        json.dump(val_buckets, f, indent=2)

    with open(run_path / "test_buckets.json", "w") as f:
        json.dump(test_buckets, f, indent=2)

    with open(run_path / "run_metadata.json", "w") as f:
        json.dump({
            "dataset_ids": dataset_ids,
            "bucket_policy": bucket_policy
        }, f, indent=2)

    print(f"Bucket snapshot saved to {run_path}")