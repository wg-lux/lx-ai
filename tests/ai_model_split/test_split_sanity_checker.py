from __future__ import annotations

import pytest

from lx_ai.ai_model_split.split_sanity_checker import verify_split_disjointness


def test_verify_split_disjointness_valid_split_passes() -> None:
    # checks valid train validation and test split does not raise error
    verify_split_disjointness(
        train_indices=[0, 1, 2],
        val_indices=[3, 4],
        test_indices=[5, 6],
    )


def test_verify_split_disjointness_allows_empty_validation_split() -> None:
    # checks empty validation split is allowed if there is no overlap
    verify_split_disjointness(
        train_indices=[0, 1, 2],
        val_indices=[],
        test_indices=[3, 4],
    )


def test_verify_split_disjointness_allows_empty_test_split() -> None:
    # checks empty test split is allowed if there is no overlap
    verify_split_disjointness(
        train_indices=[0, 1, 2],
        val_indices=[3, 4],
        test_indices=[],
    )


def test_verify_split_disjointness_allows_empty_validation_and_test_split() -> None:
    # checks empty validation and test split are allowed if train has no overlap
    verify_split_disjointness(
        train_indices=[0, 1, 2],
        val_indices=[],
        test_indices=[],
    )


def test_verify_split_disjointness_detects_train_validation_overlap() -> None:
    # checks same index cannot be in train and validation
    with pytest.raises(RuntimeError, match="Split overlap detected"):
        verify_split_disjointness(
            train_indices=[0, 1, 2],
            val_indices=[2, 3],
            test_indices=[4],
        )


def test_verify_split_disjointness_detects_train_test_overlap() -> None:
    # checks same index cannot be in train and test
    with pytest.raises(RuntimeError, match="Split overlap detected"):
        verify_split_disjointness(
            train_indices=[0, 1, 2],
            val_indices=[3],
            test_indices=[2, 4],
        )


def test_verify_split_disjointness_detects_validation_test_overlap() -> None:
    # checks same index cannot be in validation and test
    with pytest.raises(RuntimeError, match="Split overlap detected"):
        verify_split_disjointness(
            train_indices=[0, 1],
            val_indices=[2, 3],
            test_indices=[3, 4],
        )


def test_verify_split_disjointness_detects_multiple_overlaps() -> None:
    # checks function fails when more than one split pair overlaps
    with pytest.raises(RuntimeError, match="Split overlap detected"):
        verify_split_disjointness(
            train_indices=[0, 1, 2],
            val_indices=[2, 3],
            test_indices=[1, 3],
        )


def test_verify_split_disjointness_does_not_check_missing_indices() -> None:
    # checks current function only checks overlap and does not check full coverage
    verify_split_disjointness(
        train_indices=[0],
        val_indices=[2],
        test_indices=[],
    )


def test_verify_split_disjointness_does_not_detect_duplicate_inside_same_split() -> None:
    # checks current function does not detect duplicate inside same split
    # this documents current behavior and shows possible future improvement
    verify_split_disjointness(
        train_indices=[0, 1, 1],
        val_indices=[2],
        test_indices=[3],
    )
