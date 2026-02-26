#lx_ai/ai_model_split/bucket_integrity_checker.py
from __future__ import annotations

from typing import List, Optional

from lx_ai.utils.logging_utils import section, subsection, table_header


def verify_bucket_integrity(
    *,
    frame_ids: List[int],
    old_examination_ids: List[Optional[int]],
    bucket_ids: List[int],
) -> None:
    """
    Verifies:
      1) Same frame_id never assigned to different buckets
      2) Same old_examination_id never assigned to different buckets

    Uses green logging utilities.
    Raises RuntimeError if violation detected.
    """

    #section("BUCKET INTEGRITY CHECK", icon="🔒")

    # ---------------------------------------------------------
    # FRAME ID CHECK
    # ---------------------------------------------------------
    subsection("Frame ID Consistency")

    frame_to_bucket: dict[int, int] = {}
    frame_conflicts: list[tuple[int, int, int]] = []

    for fid, b in zip(frame_ids, bucket_ids):
        if fid not in frame_to_bucket:
            frame_to_bucket[fid] = b
        else:
            if frame_to_bucket[fid] != b:
                frame_conflicts.append((fid, frame_to_bucket[fid], b))

    if frame_conflicts:
        table_header("Frame ID", "Bucket A", "Bucket B")
        for fid, b1, b2 in frame_conflicts:
            print(f"{fid:<10}  {b1:<10}  {b2:<10}")
        raise RuntimeError("Frame bucket integrity violation detected.")
    else:
        print("✔ All frame_ids map to exactly one bucket.")

    # ---------------------------------------------------------
    # EXAMINATION ID CHECK
    # ---------------------------------------------------------
    subsection("Examination ID Consistency")

    exam_to_bucket: dict[int, int] = {}
    exam_conflicts: list[tuple[int, int, int]] = []

    for exam_id, b in zip(old_examination_ids, bucket_ids):
        if exam_id is None:
            continue

        if exam_id not in exam_to_bucket:
            exam_to_bucket[exam_id] = b
        else:
            if exam_to_bucket[exam_id] != b:
                exam_conflicts.append((exam_id, exam_to_bucket[exam_id], b))

    if exam_conflicts:
        table_header("Exam ID", "Bucket A", "Bucket B")
        for eid, b1, b2 in exam_conflicts:
            print(f"{eid:<10}  {b1:<10}  {b2:<10}")
        raise RuntimeError("Examination bucket integrity violation detected.")
    else:
        print("✔ All old_examination_ids map to exactly one bucket.")

    print()
