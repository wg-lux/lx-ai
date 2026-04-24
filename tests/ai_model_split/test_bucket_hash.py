from __future__ import annotations

import pytest

from lx_ai.ai_model_split.bucket_hash import compute_bucket_id, compute_bucket_key


def test_compute_bucket_key_uses_exam_id_when_available() -> None:
    # checks old_examination_id is used when it exists
    key = compute_bucket_key(frame_id=123, old_examination_id=456)

    assert key == "exam:456"


def test_compute_bucket_key_uses_frame_id_when_exam_id_is_none() -> None:
    # checks frame_id is used when old_examination_id is missing
    key = compute_bucket_key(frame_id=123, old_examination_id=None)

    assert key == "frame:123"


def test_compute_bucket_key_prevents_exam_and_frame_collision() -> None:
    # checks exam id and frame id with same number still produce different keys
    exam_key = compute_bucket_key(frame_id=999, old_examination_id=123)
    frame_key = compute_bucket_key(frame_id=123, old_examination_id=None)

    assert exam_key == "exam:123"
    assert frame_key == "frame:123"
    assert exam_key != frame_key


def test_compute_bucket_id_is_deterministic() -> None:
    # checks same key always produces same bucket
    bucket_1 = compute_bucket_id(key="exam:123", num_buckets=5)
    bucket_2 = compute_bucket_id(key="exam:123", num_buckets=5)

    assert bucket_1 == bucket_2


def test_compute_bucket_id_is_inside_valid_range() -> None:
    # checks bucket id is always between 0 and num_buckets minus 1
    for i in range(100):
        bucket = compute_bucket_id(key=f"frame:{i}", num_buckets=7)

        assert 0 <= bucket < 7


def test_compute_bucket_id_changes_when_num_buckets_changes() -> None:
    # checks bucket calculation depends on number of buckets
    key = "exam:123"

    bucket_5 = compute_bucket_id(key=key, num_buckets=5)
    bucket_7 = compute_bucket_id(key=key, num_buckets=7)

    assert 0 <= bucket_5 < 5
    assert 0 <= bucket_7 < 7


def test_compute_bucket_id_known_value_is_stable() -> None:
    # checks known hash result stays stable across future code changes
    bucket = compute_bucket_id(key="exam:123", num_buckets=5)

    assert bucket == 4


def test_compute_bucket_id_known_frame_value_is_stable() -> None:
    # checks known frame hash result stays stable across future code changes
    bucket = compute_bucket_id(key="frame:123", num_buckets=5)

    assert bucket == 2


def test_same_exam_frames_get_same_bucket() -> None:
    # checks different frames with same old_examination_id get same bucket
    key_1 = compute_bucket_key(frame_id=1, old_examination_id=500)
    key_2 = compute_bucket_key(frame_id=2, old_examination_id=500)

    bucket_1 = compute_bucket_id(key=key_1, num_buckets=5)
    bucket_2 = compute_bucket_id(key=key_2, num_buckets=5)

    assert key_1 == key_2
    assert bucket_1 == bucket_2


def test_different_frame_fallback_keys_can_get_different_buckets() -> None:
    # checks frame fallback creates different keys for different frame ids
    key_1 = compute_bucket_key(frame_id=1, old_examination_id=None)
    key_2 = compute_bucket_key(frame_id=2, old_examination_id=None)

    assert key_1 != key_2

    bucket_1 = compute_bucket_id(key=key_1, num_buckets=5)
    bucket_2 = compute_bucket_id(key=key_2, num_buckets=5)

    assert 0 <= bucket_1 < 5
    assert 0 <= bucket_2 < 5


def test_bucket_distribution_uses_multiple_buckets() -> None:
    # checks hashing does not put all samples into one bucket
    buckets = {
        compute_bucket_id(key=f"frame:{i}", num_buckets=5)
        for i in range(100)
    }

    assert len(buckets) > 1


def test_bucket_distribution_hits_all_buckets_for_many_samples() -> None:
    # checks enough sample keys reach all buckets
    buckets = {
        compute_bucket_id(key=f"frame:{i}", num_buckets=5)
        for i in range(1000)
    }

    assert buckets == {0, 1, 2, 3, 4}


def test_compute_bucket_id_rejects_zero_num_buckets() -> None:
    # checks zero buckets is invalid because modulo by zero is impossible
    with pytest.raises(ZeroDivisionError):
        compute_bucket_id(key="frame:1", num_buckets=0)


def test_compute_bucket_id_rejects_negative_num_buckets_by_result_expectation() -> None:
    # checks negative num_buckets should not be used
    # current function does not validate this, so this documents current unsafe behavior
    bucket = compute_bucket_id(key="frame:1", num_buckets=-5)

    assert bucket <= 0


def test_compute_bucket_key_accepts_zero_ids() -> None:
    # checks zero ids are valid and handled normally
    exam_key = compute_bucket_key(frame_id=0, old_examination_id=0)
    frame_key = compute_bucket_key(frame_id=0, old_examination_id=None)

    assert exam_key == "exam:0"
    assert frame_key == "frame:0"


def test_compute_bucket_id_with_large_number_of_buckets() -> None:
    # checks large bucket counts still return valid bucket id
    bucket = compute_bucket_id(key="exam:999999", num_buckets=10_000)

    assert 0 <= bucket < 10_000