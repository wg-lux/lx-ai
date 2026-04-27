# lx_ai/ai_model_split/bucket_integrity_checker.py
from __future__ import annotations

from typing import List, Optional

from lx_ai.utils.logging_utils import subsection, table_header


def verify_bucket_integrity(
    *,
    frame_ids: List[int],
    old_examination_ids: List[Optional[int]],
    bucket_ids: List[int],
    video_ids: List[int] | None = None,
) -> None:
    """
    Verifies:
      1) Same frame_id never assigned to different buckets
      2) Same old_examination_id never assigned to different buckets
      3) Same video_id never assigned to different buckets (if provided)

    Raises RuntimeError on violation.
    """

    if not (len(frame_ids) == len(old_examination_ids) == len(bucket_ids)):
        raise ValueError("frame_ids, old_examination_ids, bucket_ids must have same length")

    if video_ids is not None and len(video_ids) != len(bucket_ids):
        raise ValueError("video_ids must align with bucket_ids")

    # ---------------------------------------------------------
    # FRAME ID CHECK
    # ---------------------------------------------------------
    subsection("Frame ID Consistency")

    frame_to_bucket: dict[int, int] = {}
    frame_conflicts: list[tuple[int, int, int]] = []

    for fid, b in zip(frame_ids, bucket_ids):
        if fid not in frame_to_bucket:
            frame_to_bucket[fid] = b
        elif frame_to_bucket[fid] != b:
            frame_conflicts.append((fid, frame_to_bucket[fid], b))

    if frame_conflicts:
        table_header("Frame ID", "Bucket A", "Bucket B")
        for fid, b1, b2 in frame_conflicts:
            print(f"{fid:<10}  {b1:<10}  {b2:<10}")
        raise RuntimeError("Frame bucket integrity violation detected.")
    else:
        print(" All frame_ids map to exactly one bucket.")

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
        elif exam_to_bucket[exam_id] != b:
            exam_conflicts.append((exam_id, exam_to_bucket[exam_id], b))

    if exam_conflicts:
        table_header("Exam ID", "Bucket A", "Bucket B")
        for eid, b1, b2 in exam_conflicts:
            print(f"{eid:<10}  {b1:<10}  {b2:<10}")
        raise RuntimeError("Examination bucket integrity violation detected.")
    else:
        print(" All old_examination_ids map to exactly one bucket.")

    # ---------------------------------------------------------
    # VIDEO ID CHECK (NEW)
    # ---------------------------------------------------------
    if video_ids is not None:
        subsection("Video ID Consistency")

        video_to_bucket: dict[int, int] = {}
        video_conflicts: list[tuple[int, int, int]] = []

        for vid, b in zip(video_ids, bucket_ids):
            if vid not in video_to_bucket:
                video_to_bucket[vid] = b
            elif video_to_bucket[vid] != b:
                video_conflicts.append((vid, video_to_bucket[vid], b))

        if video_conflicts:
            table_header("Video ID", "Bucket A", "Bucket B")
            for vid, b1, b2 in video_conflicts:
                print(f"{vid:<10}  {b1:<10}  {b2:<10}")
            raise RuntimeError("Video bucket integrity violation detected.")
        else:
            print(" All video_ids map to exactly one bucket.")

    print()