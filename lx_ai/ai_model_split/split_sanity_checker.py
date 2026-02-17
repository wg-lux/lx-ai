# lx_ai/ai_model_split/split_sanity_checker.py

from lx_ai.utils.logging_utils import section, subsection, table_header


def verify_split_disjointness(
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
) -> None:
    #section("SPLIT SANITY CHECK", "🧪")

    subsection("Split Disjointness Check")

    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    if overlap_tv or overlap_tt or overlap_vt:
        table_header("Split Pair", "Overlap Count")

        if overlap_tv:
            print(f"{'Train-Val':<15} {len(overlap_tv)}")
        if overlap_tt:
            print(f"{'Train-Test':<15} {len(overlap_tt)}")
        if overlap_vt:
            print(f"{'Val-Test':<15} {len(overlap_vt)}")

        raise RuntimeError("Split overlap detected.")

    print("✔ Splits are mutually exclusive.")
