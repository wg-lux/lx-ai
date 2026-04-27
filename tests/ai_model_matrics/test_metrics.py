from __future__ import annotations

import pytest
import torch

from lx_ai.ai_model_matrics.metrics import (
    compute_metrics,
    compute_pos_only_metrics,
    compute_pos_only_metrics_per_label,
)


def test_compute_metrics_all_predictions_correct() -> None:
    # checks normal metrics when all predictions are correct
    logits = torch.tensor(
        [
            [5.0, -5.0],
            [-5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=torch.float32,
    )
    masks = torch.ones_like(targets)

    result = compute_metrics(logits, targets, masks)

    assert result["tp"] == 2
    assert result["tn"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] == pytest.approx(1.0, abs=1e-5)
    assert result["recall"] == pytest.approx(1.0, abs=1e-5)
    assert result["f1"] == pytest.approx(1.0, abs=1e-5)
    assert result["accuracy"] == pytest.approx(1.0, abs=1e-5)


def test_compute_metrics_counts_false_positive_and_false_negative() -> None:
    # checks tp fp tn fn counts when predictions are mixed
    logits = torch.tensor(
        [
            [5.0, 5.0],
            [-5.0, -5.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=torch.float32,
    )
    masks = torch.ones_like(targets)

    result = compute_metrics(logits, targets, masks)

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["tn"] == 1
    assert result["fn"] == 1


def test_compute_metrics_ignores_masked_unknown_positions() -> None:
    # checks mask 0 positions are ignored completely
    logits = torch.tensor(
        [
            [5.0, 5.0],
            [-5.0, -5.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=torch.float32,
    )
    masks = torch.tensor(
        [
            [1, 0],
            [1, 0],
        ],
        dtype=torch.float32,
    )

    result = compute_metrics(logits, targets, masks)

    assert result["tp"] == 1
    assert result["tn"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["accuracy"] == pytest.approx(1.0, abs=1e-5)


def test_compute_metrics_per_label_values() -> None:
    # checks per label metrics are calculated for each label column
    logits = torch.tensor(
        [
            [5.0, -5.0],
            [5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=torch.float32,
    )
    masks = torch.ones_like(targets)

    result = compute_metrics(logits, targets, masks)

    assert len(result["per_label"]) == 2

    label_0 = result["per_label"][0]
    label_1 = result["per_label"][1]

    assert label_0["support"] == 1
    assert label_1["support"] == 1

    assert label_0["precision"] == pytest.approx(0.5, abs=1e-5)
    assert label_0["recall"] == pytest.approx(1.0, abs=1e-5)

    assert label_1["precision"] == pytest.approx(1.0, abs=1e-5)
    assert label_1["recall"] == pytest.approx(1.0, abs=1e-5)


def test_compute_metrics_per_label_returns_none_when_no_valid_entries() -> None:
    # checks per label metric becomes None when label has no known entries
    logits = torch.tensor(
        [
            [5.0, -5.0],
            [-5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=torch.float32,
    )
    masks = torch.tensor(
        [
            [1, 0],
            [1, 0],
        ],
        dtype=torch.float32,
    )

    result = compute_metrics(logits, targets, masks)

    assert result["per_label"][1]["precision"] is None
    assert result["per_label"][1]["recall"] is None
    assert result["per_label"][1]["f1"] is None
    assert result["per_label"][1]["support"] == 0


def test_compute_metrics_rejects_invalid_threshold() -> None:
    # checks threshold must be between 0 and 1
    logits = torch.zeros((2, 2), dtype=torch.float32)
    targets = torch.zeros((2, 2), dtype=torch.float32)
    masks = torch.ones((2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="threshold must be in"):
        compute_metrics(logits, targets, masks, threshold=1.5)


def test_compute_metrics_rejects_non_2d_logits() -> None:
    # checks logits must be 2D
    logits = torch.zeros((2, 2, 2), dtype=torch.float32)
    targets = torch.zeros((2, 2, 2), dtype=torch.float32)
    masks = torch.ones((2, 2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="logits must be 2D"):
        compute_metrics(logits, targets, masks)


def test_compute_metrics_rejects_target_shape_mismatch() -> None:
    # checks targets shape must match logits shape
    logits = torch.zeros((2, 3), dtype=torch.float32)
    targets = torch.zeros((2, 2), dtype=torch.float32)
    masks = torch.ones((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="targets must match logits shape"):
        compute_metrics(logits, targets, masks)


def test_compute_metrics_rejects_mask_shape_mismatch() -> None:
    # checks masks shape must match logits shape
    logits = torch.zeros((2, 3), dtype=torch.float32)
    targets = torch.zeros((2, 3), dtype=torch.float32)
    masks = torch.ones((2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="masks must match logits shape"):
        compute_metrics(logits, targets, masks)


def test_pos_only_metrics_returns_recall_and_mean_probability() -> None:
    # checks positives only metric for known positive positions
    logits = torch.tensor(
        [
            [5.0, -5.0],
            [5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [1, 0],
            [1, 1],
        ],
        dtype=torch.float32,
    )
    masks = torch.tensor(
        [
            [1, 0],
            [1, 1],
        ],
        dtype=torch.float32,
    )

    result = compute_pos_only_metrics(logits, targets, masks)

    assert result["num_pos"] == 3
    assert result["recall_pos"] == pytest.approx(1.0, abs=1e-5)
    assert result["mean_prob_pos"] > 0.9


def test_pos_only_metrics_returns_zero_when_no_known_positives() -> None:
    # checks positives only metric returns zero values when no positives exist
    logits = torch.zeros((2, 2), dtype=torch.float32)
    targets = torch.zeros((2, 2), dtype=torch.float32)
    masks = torch.ones((2, 2), dtype=torch.float32)

    result = compute_pos_only_metrics(logits, targets, masks)

    assert result["num_pos"] == 0
    assert result["recall_pos"] == 0.0
    assert result["mean_prob_pos"] == 0.0


def test_pos_only_metrics_ignores_unknown_positions() -> None:
    # checks unknown positions do not count in positives only metric
    logits = torch.tensor([[5.0, 5.0]], dtype=torch.float32)
    targets = torch.tensor([[1, 1]], dtype=torch.float32)
    masks = torch.tensor([[1, 0]], dtype=torch.float32)

    result = compute_pos_only_metrics(logits, targets, masks)

    assert result["num_pos"] == 1
    assert result["recall_pos"] == pytest.approx(1.0, abs=1e-5)


def test_pos_only_metrics_per_label_returns_one_row_per_label() -> None:
    # checks positives only per label result has one entry per label
    logits = torch.tensor(
        [
            [5.0, -5.0],
            [5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [1, 0],
            [1, 1],
        ],
        dtype=torch.float32,
    )
    masks = torch.tensor(
        [
            [1, 0],
            [1, 1],
        ],
        dtype=torch.float32,
    )

    result = compute_pos_only_metrics_per_label(logits, targets, masks)

    assert len(result["per_label"]) == 2

    assert result["per_label"][0]["positive_support"] == 2
    assert result["per_label"][0]["known_count"] == 2
    assert result["per_label"][0]["unknown_count"] == 0
    assert result["per_label"][0]["recall_pos"] == pytest.approx(1.0, abs=1e-5)

    assert result["per_label"][1]["positive_support"] == 1
    assert result["per_label"][1]["known_count"] == 1
    assert result["per_label"][1]["unknown_count"] == 1
    assert result["per_label"][1]["recall_pos"] == pytest.approx(1.0, abs=1e-5)


def test_pos_only_metrics_per_label_returns_none_when_label_has_no_positive_support() -> None:
    # checks per label positives only metric returns None when label has no known positives
    logits = torch.zeros((2, 2), dtype=torch.float32)
    targets = torch.tensor(
        [
            [1, 0],
            [1, 0],
        ],
        dtype=torch.float32,
    )
    masks = torch.ones_like(targets)

    result = compute_pos_only_metrics_per_label(logits, targets, masks)

    assert result["per_label"][1]["positive_support"] == 0
    assert result["per_label"][1]["recall_pos"] is None
    assert result["per_label"][1]["mean_prob_pos"] is None


def test_pos_only_metrics_rejects_shape_mismatch() -> None:
    # checks positives only metrics reject invalid shapes
    logits = torch.zeros((2, 3), dtype=torch.float32)
    targets = torch.zeros((2, 2), dtype=torch.float32)
    masks = torch.ones((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="targets must match logits shape"):
        compute_pos_only_metrics(logits, targets, masks)


def test_pos_only_per_label_metrics_rejects_shape_mismatch() -> None:
    # checks positives only per label metrics reject invalid shapes
    logits = torch.zeros((2, 3), dtype=torch.float32)
    targets = torch.zeros((2, 2), dtype=torch.float32)
    masks = torch.ones((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="targets must match logits shape"):
        compute_pos_only_metrics_per_label(logits, targets, masks)