# lx_ai/utils/bucket_utils.py
import hashlib
from typing import List, Dict

def stable_hash(key: str) -> int:
    """
    A stable hash function that generates a deterministic hash based on a given key.
    This hash function will ensure that a key always maps to the same bucket, regardless of environment.
    """
    hash_object = hashlib.sha256(key.encode('utf-8'))
    return int(hash_object.hexdigest(), 16)

def get_hash_key(data: dict) -> str:
    """
    Determines the key for hashing based on the presence of `old_examination_id`.
    First, check if `old_examination_id` exists. If yes, use it. Otherwise, use `frame_id`.
    """
    if 'old_examination_id' in data and data['old_examination_id']:
        return str(data['old_examination_id'])  # If old_examination_id exists, use it for the hash
    else:
        return str(data['frame_id'])  # If not, fall back to frame_id

def assign_bucket(data: dict, num_buckets: int, bucket_mapping: Dict[str, int]) -> int:
    """
    Assign a sample to a specific bucket using the stable hash function.
    If `old_examination_id` exists, check if any frames from the same examination are assigned to a bucket.
    Otherwise, use frame_id to assign it to a bucket.
    """
    key = get_hash_key(data)  # Use either frame_id or old_examination_id to generate the key
    
    # If the key already exists, assign it to the same bucket
    if key in bucket_mapping:
        return bucket_mapping[key]
    
    # Otherwise, create a new bucket assignment using the hash
    bucket_id = stable_hash(key) % num_buckets
    bucket_mapping[key] = bucket_id  # Save the assignment for future consistency
    
    return bucket_id

def split_data_into_buckets(data: List[dict], num_buckets: int) -> Dict[int, List[dict]]:
    """
    Splits the given data into buckets using stable hash function.
    The data must be in the form of a list of dictionaries with either `old_examination_id` or `frame_id`.
    """
    buckets = {i: [] for i in range(num_buckets)}  # Create empty lists for each bucket
    bucket_mapping = {}  # To track bucket assignments for `old_examination_id` or `frame_id`
    
    for item in data:
        bucket_id = assign_bucket(item, num_buckets, bucket_mapping)
        buckets[bucket_id].append(item)
    
    return buckets
