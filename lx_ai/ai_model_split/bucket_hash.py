# lx_ai/ai_model_split/bucket_hash.py
from __future__ import annotations

import hashlib
_DEBUG_PRINT_COUNT = 0
_DEBUG_PRINT_LIMIT = 2


def compute_bucket_id(*, key: str, num_buckets: int) -> int:
    """
    Stable SHA256 hash to bucket_id in [0, num_buckets-1].
    Deterministic across machines and time.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    global _DEBUG_PRINT_COUNT
    
    if _DEBUG_PRINT_COUNT < _DEBUG_PRINT_LIMIT:
        print("For Example : \n")
        print(f"Key '{key}' hashed to digest {digest.hex()}")
        _DEBUG_PRINT_COUNT += 1
    
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
