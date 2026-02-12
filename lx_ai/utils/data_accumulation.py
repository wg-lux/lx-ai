# lx_ai/utils/data_accumulation.py
from typing import List
import json
from lx_ai.utils.db_loader_for_model_input import load_annotations_from_postgres


'''def load_datasets(dataset_ids: List[int]) -> List[dict]:
    """
    Load multiple datasets based on their dataset IDs.
    Returns a list of data entries from all datasets.
    """
    datasets = []
    for dataset_id in dataset_ids:
        # Load dataset from Postgres or JSONL
        data = load_data_from_postgres(dataset_id)
        datasets.extend(data)  # Combine all datasets
    return datasets'''

'''def load_datasets(dataset_ids: List[int]) -> List[dict]:
    """
    Load multiple datasets based on their dataset IDs.
    Returns a list of data entries from all datasets.
    """
    datasets = []
    for dataset_id in dataset_ids:
        # Load dataset from Postgres or JSONL
        data = load_annotations_from_postgres(dataset_id)  # Use real function here
        datasets.extend(data)  # Combine all datasets
    return datasets'''
def load_datasets(dataset_ids: List[int]) -> List[dict]:
    """
    Load multiple datasets based on their dataset IDs.
    Returns a list of data entries from all datasets.
    """
    datasets = load_annotations_from_postgres(dataset_ids)  # Now calling the updated function
    return datasets



def merge_datasets(datasets: List[dict]) -> List[dict]:
    """
    Merges multiple datasets into a single dataset without duplicating data.
    """
    seen = set()
    merged_data = []
    
    for dataset in datasets:
        for item in dataset:
            # Ensure 'frame', 'label', and 'value' keys exist
            frame = item.get("frame")
            label = item.get("label")
            value = item.get("value")

            # Skip if any required fields are missing
            if frame is None or label is None or value is None:
                print(f"Warning: Skipping invalid data item: {item}")
                continue

            # Ensure frame.id is unique before adding it to merged_data
            if frame["id"] not in seen:
                merged_data.append(item)
                seen.add(frame["id"])

    # Print the first 5 to 10 entries of merged data to verify structure
    print("Sample data preview (first 5 entries):")
    for i, data_item in enumerate(merged_data[:10]):
        print(f"Entry {i+1}: {data_item}")

    return merged_data

