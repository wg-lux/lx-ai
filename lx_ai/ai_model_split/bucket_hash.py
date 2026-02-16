from __future__ import annotations

import hashlib


def compute_bucket_id(*, key: str, num_buckets: int) -> int:
    """
    Stable SHA256 hash → bucket_id in [0, num_buckets-1].
    Deterministic across machines and time.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest, "big")
    return value % num_buckets


def compute_bucket_key(*, frame_id: int, old_examination_id: int | None) -> str:
    """
    Key rules (your leakage prevention requirement):
    - If old_examination_id exists → all frames in same exam share one bucket.
    - Else → fall back to frame_id.
    Prefix prevents exam_id=123 and frame_id=123 from colliding as keys.
    """
    if old_examination_id is not None:
        return f"exam:{old_examination_id}"
    return f"frame:{frame_id}"
