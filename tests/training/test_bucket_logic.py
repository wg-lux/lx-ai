from __future__ import annotations

import pytest

from lx_ai.training.bucket_logic import build_bucket_key, compute_bucket


class TestBucketLogic:
    def test_build_bucket_key_uses_exam_id_when_available(self) -> None:
        # checks old_examination_id is used when it exists
        key = build_bucket_key(frame_id=123, old_examination_id=456)

        assert key == "exam:456"

    def test_build_bucket_key_uses_frame_id_when_exam_id_is_none(self) -> None:
        # checks frame_id is used when old_examination_id is missing
        key = build_bucket_key(frame_id=123, old_examination_id=None)

        assert key == "frame:123"

    def test_build_bucket_key_prevents_exam_and_frame_collision(self) -> None:
        # checks exam id and frame id with same number still produce different keys
        exam_key = build_bucket_key(frame_id=999, old_examination_id=123)
        frame_key = build_bucket_key(frame_id=123, old_examination_id=None)

        assert exam_key == "exam:123"
        assert frame_key == "frame:123"
        assert exam_key != frame_key

    def test_build_bucket_key_accepts_zero_exam_id(self) -> None:
        # checks old_examination_id zero is treated as real exam id
        key = build_bucket_key(frame_id=123, old_examination_id=0)

        assert key == "exam:0"

    def test_build_bucket_key_accepts_zero_frame_id(self) -> None:
        # checks frame_id zero is handled correctly
        key = build_bucket_key(frame_id=0, old_examination_id=None)

        assert key == "frame:0"

    def test_compute_bucket_is_deterministic(self) -> None:
        # checks same key always gives same bucket
        bucket_1 = compute_bucket("exam:123", 5)
        bucket_2 = compute_bucket("exam:123", 5)

        assert bucket_1 == bucket_2

    def test_compute_bucket_returns_value_inside_range(self) -> None:
        # checks bucket result is between 0 and num_buckets minus 1
        for i in range(100):
            bucket = compute_bucket(f"frame:{i}", 7)

            assert 0 <= bucket < 7

    def test_compute_bucket_known_exam_value_is_stable(self) -> None:
        # checks known exam hash result stays stable across future code changes
        bucket = compute_bucket("exam:123", 5)

        assert bucket == 1

    def test_compute_bucket_known_frame_value_is_stable(self) -> None:
        # checks known frame hash result stays stable across future code changes
        bucket = compute_bucket("frame:123", 5)

        assert bucket == 0

    def test_same_exam_frames_get_same_bucket(self) -> None:
        # checks different frames with same old_examination_id get same bucket
        key_1 = build_bucket_key(frame_id=1, old_examination_id=500)
        key_2 = build_bucket_key(frame_id=2, old_examination_id=500)

        bucket_1 = compute_bucket(key_1, 5)
        bucket_2 = compute_bucket(key_2, 5)

        assert key_1 == key_2
        assert bucket_1 == bucket_2

    def test_different_frame_fallback_keys_are_different(self) -> None:
        # checks frames without exam id use their own frame id in key
        key_1 = build_bucket_key(frame_id=1, old_examination_id=None)
        key_2 = build_bucket_key(frame_id=2, old_examination_id=None)

        assert key_1 != key_2

    def test_compute_bucket_uses_multiple_buckets_for_many_keys(self) -> None:
        # checks hashing does not put all keys into one bucket
        buckets = {compute_bucket(f"frame:{i}", 5) for i in range(100)}

        assert len(buckets) > 1

    def test_compute_bucket_hits_all_buckets_for_many_keys(self) -> None:
        # checks enough keys reach all buckets
        buckets = {compute_bucket(f"frame:{i}", 5) for i in range(1000)}

        assert buckets == {0, 1, 2, 3, 4}

    def test_compute_bucket_changes_range_when_num_buckets_changes(self) -> None:
        # checks num_buckets controls result range
        key = "exam:123"

        bucket_5 = compute_bucket(key, 5)
        bucket_10 = compute_bucket(key, 10)

        assert 0 <= bucket_5 < 5
        assert 0 <= bucket_10 < 10

    def test_compute_bucket_rejects_zero_num_buckets(self) -> None:
        # checks zero buckets fails because modulo by zero is impossible
        with pytest.raises(ZeroDivisionError):
            compute_bucket("frame:1", 0)

    def test_compute_bucket_negative_num_buckets_documents_current_behavior(
        self,
    ) -> None:
        # checks current behavior for negative num_buckets
        # this is not desired for production but documents current function behavior
        bucket = compute_bucket("frame:1", -5)

        assert bucket <= 0

    def test_compute_bucket_large_num_buckets(self) -> None:
        # checks large bucket count still returns valid bucket id
        bucket = compute_bucket("exam:999999", 10_000)

        assert 0 <= bucket < 10_000
