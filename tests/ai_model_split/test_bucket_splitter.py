from __future__ import annotations

import pytest
from pydantic import ValidationError

from lx_ai.ai_model_split.bucket_splitter import (
    BucketSplitPolicy,
    split_indices_by_bucket_policy,
)


def test_bucket_split_policy_valid_config_passes() -> None:
    # checks valid bucket split policy is accepted
    policy = BucketSplitPolicy(
        num_buckets=5,
        validation_buckets=[3],
        test_buckets=[4],
    )

    assert policy.num_buckets == 5
    assert policy.validation_buckets == [3]
    assert policy.test_buckets == [4]


def test_bucket_split_policy_train_buckets_are_complement() -> None:
    # checks train buckets are all buckets except validation and test buckets
    policy = BucketSplitPolicy(
        num_buckets=5,
        validation_buckets=[3],
        test_buckets=[4],
    )

    assert policy.train_buckets == [0, 1, 2]


def test_bucket_split_policy_to_meta_returns_plain_dict() -> None:
    # checks policy can be converted to metadata dict
    policy = BucketSplitPolicy(
        num_buckets=5,
        validation_buckets=[3],
        test_buckets=[4],
    )

    meta = policy.to_meta()

    assert meta == {
        "num_buckets": 5,
        "validation_buckets": [3],
        "test_buckets": [4],
        "train_buckets": [0, 1, 2],
    }


def test_bucket_split_policy_rejects_too_few_buckets() -> None:
    # checks num_buckets must be at least 3
    with pytest.raises(ValidationError):
        BucketSplitPolicy(
            num_buckets=2,
            validation_buckets=[0],
            test_buckets=[1],
        )


def test_bucket_split_policy_rejects_multiple_validation_buckets() -> None:
    # checks only one validation bucket is allowed
    with pytest.raises(ValidationError, match="validation_buckets must contain exactly one"):
        BucketSplitPolicy(
            num_buckets=5,
            validation_buckets=[2, 3],
            test_buckets=[4],
        )


def test_bucket_split_policy_rejects_multiple_test_buckets() -> None:
    # checks only one test bucket is allowed
    with pytest.raises(ValidationError, match="test_buckets must contain exactly one"):
        BucketSplitPolicy(
            num_buckets=5,
            validation_buckets=[3],
            test_buckets=[1, 4],
        )


def test_bucket_split_policy_rejects_duplicate_validation_bucket_list_as_multiple_buckets() -> None:
    # checks duplicate validation bucket list is rejected because only one validation bucket is allowed
    with pytest.raises(ValidationError, match="validation_buckets must contain exactly one"):
        BucketSplitPolicy(
            num_buckets=5,
            validation_buckets=[3, 3],
            test_buckets=[4],
        )


def test_bucket_split_policy_rejects_duplicate_test_bucket_list_as_multiple_buckets() -> None:
    # checks duplicate test bucket list is rejected because only one test bucket is allowed
    with pytest.raises(ValidationError, match="test_buckets must contain exactly one"):
        BucketSplitPolicy(
            num_buckets=5,
            validation_buckets=[3],
            test_buckets=[4, 4],
        )


def test_bucket_split_policy_rejects_out_of_range_validation_bucket() -> None:
    # checks validation bucket id must be inside bucket range
    with pytest.raises(ValidationError, match="validation_buckets contains out-of-range"):
        BucketSplitPolicy(
            num_buckets=5,
            validation_buckets=[5],
            test_buckets=[4],
        )


def test_bucket_split_policy_rejects_out_of_range_test_bucket() -> None:
    # checks test bucket id must be inside bucket range
    with pytest.raises(ValidationError, match="test_buckets contains out-of-range"):
        BucketSplitPolicy(
            num_buckets=5,
            validation_buckets=[3],
            test_buckets=[5],
        )


def test_bucket_split_policy_rejects_validation_test_overlap() -> None:
    # checks validation and test bucket cannot be same bucket
    with pytest.raises(ValidationError, match="must not overlap"):
        BucketSplitPolicy(
            num_buckets=5,
            validation_buckets=[3],
            test_buckets=[3],
        )


def test_split_indices_by_bucket_policy_returns_all_expected_outputs() -> None:
    # checks split function returns indices, bucket ids, bucket sizes and role sizes
    policy = BucketSplitPolicy(
        num_buckets=5,
        validation_buckets=[3],
        test_buckets=[4],
    )

    result = split_indices_by_bucket_policy(
        frame_ids=[1, 2, 3, 4, 5],
        old_examination_ids=[None, None, None, None, None],
        policy=policy,
    )

    train_idx, val_idx, test_idx, bucket_ids, bucket_sizes, role_sizes = result

    assert isinstance(train_idx, list)
    assert isinstance(val_idx, list)
    assert isinstance(test_idx, list)
    assert isinstance(bucket_ids, list)
    assert isinstance(bucket_sizes, dict)
    assert isinstance(role_sizes, dict)

    assert len(bucket_ids) == 5
    assert role_sizes["train"] + role_sizes["val"] + role_sizes["test"] == 5


def test_split_indices_by_bucket_policy_is_deterministic() -> None:
    # checks same input gives same split every time
    policy = BucketSplitPolicy(
        num_buckets=5,
        validation_buckets=[3],
        test_buckets=[4],
    )

    kwargs = {
        "frame_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "old_examination_ids": [None, None, 10, 10, None, 20, 20, None],
        "policy": policy,
    }

    result_1 = split_indices_by_bucket_policy(**kwargs)
    result_2 = split_indices_by_bucket_policy(**kwargs)

    assert result_1 == result_2


def test_split_indices_by_bucket_policy_covers_all_indices_once() -> None:
    # checks every sample index appears exactly once in train validation or test
    policy = BucketSplitPolicy(
        num_buckets=5,
        validation_buckets=[3],
        test_buckets=[4],
    )

    train_idx, val_idx, test_idx, _, _, _ = split_indices_by_bucket_policy(
        frame_ids=list(range(20)),
        old_examination_ids=[None] * 20,
        policy=policy,
    )

    all_split_indices = train_idx + val_idx + test_idx

    assert sorted(all_split_indices) == list(range(20))
    assert len(all_split_indices) == len(set(all_split_indices))


def test_split_indices_by_bucket_policy_keeps_same_old_exam_together() -> None:
    # checks frames with same old_examination_id get same bucket
    policy = BucketSplitPolicy(
        num_buckets=5,
        validation_buckets=[3],
        test_buckets=[4],
    )

    frame_ids = [1, 2, 3, 4]
    old_exam_ids = [100, 100, 200, 200]

    _, _, _, bucket_ids, _, _ = split_indices_by_bucket_policy(
        frame_ids=frame_ids,
        old_examination_ids=old_exam_ids,
        policy=policy,
    )

    assert bucket_ids[0] == bucket_ids[1]
    assert bucket_ids[2] == bucket_ids[3]


def test_split_indices_by_bucket_policy_uses_frame_id_when_exam_id_is_none() -> None:
    # checks frames without old_examination_id are bucketed by frame id
    policy = BucketSplitPolicy(
        num_buckets=5,
        validation_buckets=[3],
        test_buckets=[4],
    )

    frame_ids = [1, 1, 2]
    old_exam_ids = [None, None, None]

    _, _, _, bucket_ids, _, _ = split_indices_by_bucket_policy(
        frame_ids=frame_ids,
        old_examination_ids=old_exam_ids,
        policy=policy,
    )

    assert bucket_ids[0] == bucket_ids[1]


def test_split_indices_by_bucket_policy_rejects_length_mismatch() -> None:
    # checks frame_ids and old_examination_ids must have same length
    policy = BucketSplitPolicy(
        num_buckets=5,
        validation_buckets=[3],
        test_buckets=[4],
    )

    with pytest.raises(ValueError, match="must have same length"):
        split_indices_by_bucket_policy(
            frame_ids=[1, 2, 3],
            old_examination_ids=[None, None],
            policy=policy,
        )


def test_split_indices_by_bucket_policy_role_sizes_match_split_lengths() -> None:
    # checks role_sizes match actual split list lengths
    policy = BucketSplitPolicy(
        num_buckets=5,
        validation_buckets=[3],
        test_buckets=[4],
    )

    train_idx, val_idx, test_idx, _, _, role_sizes = split_indices_by_bucket_policy(
        frame_ids=list(range(30)),
        old_examination_ids=[None] * 30,
        policy=policy,
    )

    assert role_sizes["train"] == len(train_idx)
    assert role_sizes["val"] == len(val_idx)
    assert role_sizes["test"] == len(test_idx)


def test_split_indices_by_bucket_policy_bucket_sizes_match_bucket_ids() -> None:
    # checks bucket_sizes count the produced bucket ids correctly
    policy = BucketSplitPolicy(
        num_buckets=5,
        validation_buckets=[3],
        test_buckets=[4],
    )

    _, _, _, bucket_ids, bucket_sizes, _ = split_indices_by_bucket_policy(
        frame_ids=list(range(30)),
        old_examination_ids=[None] * 30,
        policy=policy,
    )

    total_from_bucket_sizes = sum(bucket_sizes.values())

    assert total_from_bucket_sizes == len(bucket_ids)
