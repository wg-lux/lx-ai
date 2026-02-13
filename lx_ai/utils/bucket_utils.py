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

def assign_bucket(data: dict, num_buckets: int, bucket_mapping: Dict[str, int], examination_mapping: Dict[str, int]) -> int:
    """
    Assign a sample to a specific bucket using the stable hash function.
    If `old_examination_id` exists, check if any frames from the same examination are assigned to a bucket.
    Otherwise, use frame_id to assign it to a bucket.
    """
    key = get_hash_key(data)  # Use either frame_id or old_examination_id to generate the key
    
    # If the key already exists in the bucket mapping, return the same bucket
    if key in bucket_mapping:
        return bucket_mapping[key]
    
    # If old_examination_id exists, check if we have an existing assignment
    old_exam_id = data.get("old_examination_id")
    if old_exam_id is not None:
        if old_exam_id in examination_mapping:
            # Assign to the same bucket as the existing examination ID
            bucket_id = examination_mapping[old_exam_id]
        else:
            # If it's a new examination_id, assign a new bucket
            bucket_id = stable_hash(key) % num_buckets
            examination_mapping[old_exam_id] = bucket_id
    else:
        # Otherwise, just hash based on frame_id
        bucket_id = stable_hash(key) % num_buckets
    
    # Store the bucket assignment for future consistency
    bucket_mapping[key] = bucket_id
    
    return bucket_id

def split_data_into_buckets(data, num_buckets):
    """
    Splits the dataset into buckets based on frame_id and old_examination_id.
    """
    buckets = {i: [] for i in range(num_buckets)}
    bucket_mapping = {}  # Store bucket assignments for each frame/old_examination_id
    examination_mapping = {}  # Store bucket assignments for each old_examination_id
    
    for item in data:
        bucket_id = assign_bucket(item, num_buckets, bucket_mapping, examination_mapping)
        buckets[bucket_id].append(item)
    
    # Now, split the buckets into train, validation, and test sets (adjust according to your logic)
    train_data = []
    val_data = []
    test_data = []
    
    # Assuming a 80-10-10 split (adjust as necessary)
    for bucket_id, bucket in buckets.items():
        train_size = int(0.8 * len(bucket))
        val_size = int(0.1 * len(bucket))
        
        train_data.extend(bucket[:train_size])
        val_data.extend(bucket[train_size:train_size + val_size])
        test_data.extend(bucket[train_size + val_size:])
        
    # Ensure it returns exactly three values
    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }
