#lx_ai/training/bucket_logic.py
import hashlib
from typing import Union


def build_bucket_key(frame_id: int,
                     old_examination_id: Union[int, None]) -> str:
    """
    Build deterministic key for hashing.
    Ensures no collision between exam and frame.
    """
    if old_examination_id is not None:
        return f"exam:{old_examination_id}"
    return f"frame:{frame_id}"


def compute_bucket(key: str, num_buckets: int) -> int:
    """
    Deterministic SHA256 hashing → bucket id.
    """
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest, "big") % num_buckets