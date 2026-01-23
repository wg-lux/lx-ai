"""
Tests for multi-label classification metrics.

This test file validates the behavior of `compute_metrics`, which is responsible
for calculating global and per-label metrics in a multi-label setting with masks.

Core ideas tested here:

1) Inputs
   - logits: raw model outputs of shape [N, C]
   - targets: ground-truth labels (0 or 1) of shape [N, C]
   - masks: validity mask (0 = ignore, 1 = valid) of shape [N, C]

2) Masking semantics (VERY IMPORTANT)
   - Metrics are computed ONLY where mask == 1
   - Masked positions behave as if they do not exist
   - This matches the "unknown label" behavior used in training

Example:
    logits  = [[10.0, -10.0]]
    targets = [[1.0,   0.0 ]]
    masks   = [[1.0,   0.0 ]]

Here:
    - Label 0 is evaluated (mask = 1)
    - Label 1 is ignored completely (mask = 0)

3) Outputs
   - Global metrics: precision, recall, f1, accuracy, tp, fp, tn, fn
   - Per-label metrics:
       * precision / recall / f1 OR None if no valid samples
       * support = number of positive samples among valid entries

These tests are intentionally small and explicit so that reading the test
also explains how the metrics logic works internally.
"""

import torch
import pytest

from lx_ai.ai_model_matrics.metrics import compute_metrics


# -----------------------------------------------------------------------------
# 1. Basic correctness
# -----------------------------------------------------------------------------

def test_compute_metrics__perfect_prediction__returns_perfect_scores() -> None:
    """
    GIVEN perfect predictions with full masks
    WHEN compute_metrics is called
    THEN all global metrics are exactly 1.0

    This verifies the happy-path behavior.
    """
    logits = torch.tensor([[10.0, -10.0]])
    targets = torch.tensor([[1.0, 0.0]])
    masks = torch.tensor([[1.0, 1.0]])

    out = compute_metrics(logits=logits, targets=targets, masks=masks)

    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(1.0)
    assert out["f1"] == pytest.approx(1.0)
    assert out["accuracy"] == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# 2. Masking semantics
# -----------------------------------------------------------------------------

def test_compute_metrics__masked_entries_are_ignored() -> None:
    """
    GIVEN partially masked labels
    WHEN compute_metrics is called
    THEN masked entries do not affect global metrics

    Example:
        masks = [1, 0] → only the first label counts
    """
    logits = torch.tensor([[10.0, 10.0]])
    targets = torch.tensor([[1.0, 0.0]])
    masks = torch.tensor([[1.0, 0.0]])

    out = compute_metrics(logits=logits, targets=targets, masks=masks)

    assert out["tp"] == 1
    assert out["fp"] == 0
    assert out["fn"] == 0
    assert out["tn"] == 0


# -----------------------------------------------------------------------------
# 3. Per-label metrics with no valid samples
# -----------------------------------------------------------------------------

def test_compute_metrics__label_with_no_valid_samples__returns_none_metrics() -> None:
    """
    GIVEN a label column that is fully masked
    WHEN compute_metrics is called
    THEN per-label precision/recall/f1 are None and support is 0

    This avoids misleading metrics for labels with no supervision.
    """
    logits = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([[0.0, 1.0]])
    masks = torch.tensor([[1.0, 0.0]])

    out = compute_metrics(logits=logits, targets=targets, masks=masks)

    per_label = out["per_label"]

    assert per_label[1]["precision"] is None
    assert per_label[1]["recall"] is None
    assert per_label[1]["f1"] is None
    assert per_label[1]["support"] == 0


# -----------------------------------------------------------------------------
# 4. Shape validation
# -----------------------------------------------------------------------------

def test_compute_metrics__non_2d_inputs__raise_error() -> None:
    """
    GIVEN invalid tensor shapes (not 2D)
    WHEN compute_metrics is called
    THEN a ValueError is raised

    This enforces strict input contracts and prevents silent bugs.
    """
    logits = torch.tensor([1.0, 2.0])
    targets = torch.tensor([1.0, 0.0])
    masks = torch.tensor([1.0, 1.0])

    with pytest.raises(ValueError):
        compute_metrics(logits=logits, targets=targets, masks=masks)
