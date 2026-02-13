# lx_ai/utils/data_accumulation.py
from typing import List
import json
from lx_ai.utils.db_loader_for_model_input import load_annotations_from_postgres

from typing import List
from lx_ai.utils.db_loader_for_model_input import load_annotations_from_postgres

def load_datasets(dataset_ids) -> List[dict]:
    print("here:",dataset_ids)
   # print(f"Type of dataset_ids: {type(dataset_ids)}")

    """
    Load multiple datasets based on their dataset IDs.
    Returns a list of data entries from all datasets.
    """
    # Ensure dataset_ids is always a list
    if isinstance(dataset_ids, int):  # If it's a single dataset ID (not a list)
        dataset_ids = [dataset_ids]  # Convert to a list
        #print(f"Type of dataset_ids after isinstance: {type(dataset_ids)}")


    datasets = []
    #print(f"Type of dataset_ids before the for loop: {type(dataset_ids)}")
    for dataset_id in dataset_ids:
        #print(f"Type of dataset_ids in teh for loop: {type(dataset_id)}")
        # Load dataset from Postgres
        if isinstance(dataset_id, int):  # If it's a single dataset ID (not a list)
            dataset_id = [dataset_id]
        #print(f"Type of dataset_id in for loop after conversion: {type(dataset_id)}")
        #exit("Terminating the script")
        data = load_annotations_from_postgres(dataset_id)


        #print(f"Data loaded for dataset_id {dataset_id}: {data[:5]}")  # Print the first 5 entries to check the format

        # Check data format here before appending to datasets
        if not isinstance(data, list):
            print(f"Warning: Loaded data is not a list")
            continue  # Skip invalid data

        datasets.append(data)  # Append each dataset

        # Print the first 5 entries to ensure correct loading
        #print(f"Loading dataset {dataset_id} from Postgres")

        #print(f"Dataset {dataset_id} loaded. Preview in load_datasets: {data[:8]}")

    # If only one dataset is loaded, no need to merge
    if len(datasets) == 1:
        print("Only one dataset loaded. No merging required.")
        return datasets[0]  # Return the single dataset

    # If multiple datasets are loaded, merge them
    merged_data = merge_datasets(datasets)
    return merged_data



import json

def merge_datasets(datasets):
    merged_data = []
    seen = set()

    # Iterate over each dataset
    for dataset in datasets:
        print(f"Processing dataset with {len(dataset)} items.")  # Debugging: check dataset length
        for item in dataset:
           # print(f"Processing item: {item}, type: {type(item)}")  # Debugging: print each item type

            # Check if item is a string or malformed data
            if isinstance(item, str):
                print(f"Warning: Invalid data item (string found): {item}")
                try:
                    # Attempt to parse the string into a dictionary
                    item = json.loads(item)
                    print(f"Successfully converted string to dict: {item}")
                except json.JSONDecodeError:
                    print(f"Failed to convert string to dict: {item}")
                    continue  # Skip this item

            # Ensure item is now a dictionary
            if not isinstance(item, dict):
                print(f"Warning: Invalid data item (not a dict): {item}")
                continue

            # Extract the 'frame', 'label', and 'value' from the item
            frame = item.get("frame")
            label = item.get("label")
            value = item.get("value")

            # Check if required keys are present
            if frame is None or label is None or value is None:
                print(f"Warning: Skipping invalid item (missing frame, label, or value): {item}")
                continue  # Skip items missing essential data

            # Ensure the 'frame_id' is unique and not already seen
            frame_id = frame.get("id")
            if frame_id:
                if frame_id not in seen:
                    merged_data.append(item)  # Add item to merged_data
                    seen.add(frame_id)  # Mark this frame_id as seen
                else:
                    print(f"Warning: Duplicate frame_id found: {frame_id}, skipping item.")
            else:
                print(f"Warning: Missing frame_id for item, skipping it: {item}")

    print(f"Merged dataset length: {len(merged_data)}")  # Debugging: print merged data length
    return merged_data

