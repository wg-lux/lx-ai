from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator
from lx_ai.utils.logging_utils import subsection
from lx_ai.ai_model_split.bucket_hash import compute_bucket_id, compute_bucket_key


class BucketSplitPolicy(BaseModel):
    """
    Immutable split policy configured in YAML.

    You want:
    - fixed num_buckets
    - exactly one validation bucket
    - exactly one test bucket
    - all other buckets are training
    - strict validation: in range, no overlaps, cover all buckets exactly once
    """
    num_buckets: int = Field(..., ge=3)
    validation_buckets: List[int] = Field(..., min_length=1)
    test_buckets: List[int] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_policy(self) -> "BucketSplitPolicy":
        # you explicitly said: one for validation, one for testing
        if len(self.validation_buckets) != 1:
            raise ValueError("validation_buckets must contain exactly one bucket id.")
        if len(self.test_buckets) != 1:
            raise ValueError("test_buckets must contain exactly one bucket id.")

        # duplicates inside lists
        if len(set(self.validation_buckets)) != len(self.validation_buckets):
            raise ValueError("validation_buckets contains duplicates.")
        if len(set(self.test_buckets)) != len(self.test_buckets):
            raise ValueError("test_buckets contains duplicates.")

        all_ids = set(range(self.num_buckets))
        val_set = set(self.validation_buckets)
        test_set = set(self.test_buckets)

        # in range
        if not val_set.issubset(all_ids):
            raise ValueError("validation_buckets contains out-of-range bucket id(s).")
        if not test_set.issubset(all_ids):
            raise ValueError("test_buckets contains out-of-range bucket id(s).")

        # no overlap
        if not val_set.isdisjoint(test_set):
            raise ValueError("validation_buckets and test_buckets must not overlap.")

        # cover all buckets exactly once:
        # training is defined as complement; so uniqueness and total coverage is guaranteed.
        return self

    @property
    def train_buckets(self) -> List[int]:
        used = set(self.validation_buckets) | set(self.test_buckets)
        return [b for b in range(self.num_buckets) if b not in used]

    def to_meta(self) -> Dict[str, Any]:
        return {
            "num_buckets": int(self.num_buckets),
            "validation_buckets": list(self.validation_buckets),
            "test_buckets": list(self.test_buckets),
            "train_buckets": self.train_buckets,
        }


def split_indices_by_bucket_policy(
    *,
    frame_ids: List[int],
    old_examination_ids: List[Optional[int]],
    policy: BucketSplitPolicy,
) -> Tuple[List[int], List[int], List[int], List[int], Dict[str, int], Dict[str, int]]:
    """
    Returns:
      - train_indices, val_indices, test_indices
      - bucket_ids_per_sample
      - bucket_sizes: {bucket_id(str): count}
      - role_sizes: {"train": n, "val": n, "test": n}
    """
    if len(frame_ids) != len(old_examination_ids):
        raise ValueError("frame_ids and old_examination_ids must have same length.")

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    bucket_ids: List[int] = []

    oldexamid_hash_count = 0
    frameid_hash_count = 0


    for i, (fid, exam_id) in enumerate(zip(frame_ids, old_examination_ids)):
        #key = compute_bucket_key(frame_id=int(fid), old_examination_id=exam_id)
        if exam_id is not None:
            oldexamid_hash_count += 1
        else:
            frameid_hash_count += 1
        
        key = compute_bucket_key(frame_id=int(fid), old_examination_id=exam_id)
        b = compute_bucket_id(key=key, num_buckets=int(policy.num_buckets))
        bucket_ids.append(b)

        if b == policy.validation_buckets[0]:
            val_idx.append(i)
        elif b == policy.test_buckets[0]:
            test_idx.append(i)
        else:
            train_idx.append(i)

    # bucket sizes
    counts = Counter(bucket_ids)
    bucket_sizes: Dict[str, int] = {str(k): int(v) for k, v in sorted(counts.items())}

    role_sizes = {
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
    }
    subsection("HASH KEY USAGE")
    print(f"  Using old_examination_id : {oldexamid_hash_count}")
    print(f"  Using frame_id fallback  : {frameid_hash_count}")
    return train_idx, val_idx, test_idx, bucket_ids, bucket_sizes, role_sizes
