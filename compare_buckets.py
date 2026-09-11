import json


# -------------------------------------------------------
# 1️⃣ Helper: Load a JSON file into a dictionary
# -------------------------------------------------------
def load_json(path):
    """
    Loads bucket assignment JSON file.
    Returns:
        dict -> { "exam:<id>": bucket_id }
    """
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------------------
# 2️⃣ Load different runs
# -------------------------------------------------------
# You can add as many runs as you want here.
run1 = load_json(
    "data/model_training/buckets/20260223_091917/all_bucket_assignments.json"
)
run2 = load_json(
    "data/model_training/buckets/20260223_093045/all_bucket_assignments.json"
)
run3 = load_json(
    "data/model_training/buckets/20260223_112622/all_bucket_assignments.json"
)


# -------------------------------------------------------
# 3️⃣ Compare two runs
# -------------------------------------------------------
def compare(reference, other, name):
    """
    Compares bucket stability between two runs.

    reference → earlier run (baseline)
    other     → later run
    name      → label for printing

    It checks:
    - For every exam in reference
    - If it also exists in 'other'
    - Whether bucket assignment stayed identical
    """

    mismatches = []
    missing = []

    for key, bucket in reference.items():
        if key not in other:
            missing.append(key)
        elif other[key] != bucket:
            mismatches.append((key, bucket, other[key]))

    print(f"\n🔎 Comparing {name}")
    print("Total exams in reference:", len(reference))
    print("Shared exams:", len(reference) - len(missing))
    print("Missing exams in other:", len(missing))
    print("Bucket mismatches:", len(mismatches))

    if mismatches:
        print("\nExamples of mismatches:")
        for m in mismatches[:10]:
            print(f"  {m[0]} → was {m[1]}, now {m[2]}")


# -------------------------------------------------------
# 4️⃣ Manual comparisons
# -------------------------------------------------------

# Dataset1 stability
compare(run1, run2, "Run2 vs Run1")
compare(run1, run3, "Run3 vs Run1")

# Dataset6 stability
compare(run2, run3, "Run3 vs Run2")
