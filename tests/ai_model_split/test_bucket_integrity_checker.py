from __future__ import annotations

import pytest

from lx_ai.ai_model_split.bucket_integrity_checker import verify_bucket_integrity


def test_bucket_integrity_valid_frame_and_exam_mapping_passes() -> None:
    # checks valid frame ids and old examination ids pass integrity check
    verify_bucket_integrity(
        frame_ids=[1, 2, 3, 4],
        old_examination_ids=[10, 10, 20, 20],
        bucket_ids=[1, 1, 2, 2],
    )


def test_bucket_integrity_valid_video_mapping_passes() -> None:
    # checks valid video ids pass integrity check
    verify_bucket_integrity(
        frame_ids=[1, 2, 3, 4],
        old_examination_ids=[10, 10, 20, 20],
        bucket_ids=[1, 1, 2, 2],
        video_ids=[100, 100, 200, 200],
    )


def test_bucket_integrity_rejects_frame_id_in_different_buckets() -> None:
    # checks same frame_id cannot be assigned to two different buckets
    with pytest.raises(RuntimeError, match="Frame bucket integrity violation"):
        verify_bucket_integrity(
            frame_ids=[1, 1],
            old_examination_ids=[None, None],
            bucket_ids=[0, 1],
        )


def test_bucket_integrity_allows_same_frame_id_same_bucket() -> None:
    # checks same frame_id is allowed when bucket is also same
    verify_bucket_integrity(
        frame_ids=[1, 1],
        old_examination_ids=[None, None],
        bucket_ids=[2, 2],
    )


def test_bucket_integrity_rejects_old_exam_id_in_different_buckets() -> None:
    # checks same old_examination_id cannot be assigned to two different buckets
    with pytest.raises(RuntimeError, match="Examination bucket integrity violation"):
        verify_bucket_integrity(
            frame_ids=[1, 2],
            old_examination_ids=[10, 10],
            bucket_ids=[0, 1],
        )


def test_bucket_integrity_allows_old_exam_id_same_bucket() -> None:
    # checks same old_examination_id is allowed when bucket is also same
    verify_bucket_integrity(
        frame_ids=[1, 2],
        old_examination_ids=[10, 10],
        bucket_ids=[3, 3],
    )


def test_bucket_integrity_ignores_none_old_exam_ids() -> None:
    # checks None old_examination_id is ignored for exam consistency check
    verify_bucket_integrity(
        frame_ids=[1, 2, 3],
        old_examination_ids=[None, None, None],
        bucket_ids=[0, 1, 2],
    )


def test_bucket_integrity_rejects_video_id_in_different_buckets() -> None:
    # checks same video_id cannot be assigned to two different buckets
    with pytest.raises(RuntimeError, match="Video bucket integrity violation"):
        verify_bucket_integrity(
            frame_ids=[1, 2],
            old_examination_ids=[None, None],
            bucket_ids=[0, 1],
            video_ids=[100, 100],
        )


def test_bucket_integrity_allows_video_id_same_bucket() -> None:
    # checks same video_id is allowed when bucket is also same
    verify_bucket_integrity(
        frame_ids=[1, 2],
        old_examination_ids=[None, None],
        bucket_ids=[4, 4],
        video_ids=[100, 100],
    )


def test_bucket_integrity_rejects_frame_exam_bucket_length_mismatch() -> None:
    # checks frame_ids old_examination_ids and bucket_ids must have same length
    with pytest.raises(ValueError, match="must have same length"):
        verify_bucket_integrity(
            frame_ids=[1, 2, 3],
            old_examination_ids=[10, 10],
            bucket_ids=[1, 1, 1],
        )


def test_bucket_integrity_rejects_video_id_length_mismatch() -> None:
    # checks video_ids must align with bucket_ids
    with pytest.raises(ValueError, match="video_ids must align"):
        verify_bucket_integrity(
            frame_ids=[1, 2, 3],
            old_examination_ids=[10, 10, 20],
            bucket_ids=[1, 1, 2],
            video_ids=[100, 100],
        )


def test_bucket_integrity_detects_frame_conflict_before_exam_conflict() -> None:
    # checks frame conflict is detected first when both frame and exam conflict exist
    with pytest.raises(RuntimeError, match="Frame bucket integrity violation"):
        verify_bucket_integrity(
            frame_ids=[1, 1],
            old_examination_ids=[10, 10],
            bucket_ids=[0, 1],
        )


def test_bucket_integrity_detects_exam_conflict_when_frame_ids_are_unique() -> None:
    # checks exam conflict is detected when frame ids themselves are unique
    with pytest.raises(RuntimeError, match="Examination bucket integrity violation"):
        verify_bucket_integrity(
            frame_ids=[1, 2, 3],
            old_examination_ids=[10, 10, 20],
            bucket_ids=[0, 1, 2],
        )


def test_bucket_integrity_detects_video_conflict_when_frame_and_exam_are_valid() -> (
    None
):
    # checks video conflict is detected after frame and exam checks pass
    with pytest.raises(RuntimeError, match="Video bucket integrity violation"):
        verify_bucket_integrity(
            frame_ids=[1, 2, 3],
            old_examination_ids=[10, 10, 20],
            bucket_ids=[0, 0, 2],
            video_ids=[100, 100, 100],
        )
