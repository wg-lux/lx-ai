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
    THEN precision, recall, f1, accuracy are all 1.0
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
    GIVEN masked label positions
    WHEN compute_metrics is called
    THEN masked entries do not affect metrics
    """
    logits = torch.tensor([[10.0, 10.0]])
    targets = torch.tensor([[1.0, 0.0]])
    masks = torch.tensor([[1.0, 0.0]])  # second label ignored

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
    GIVEN a label column fully masked
    WHEN compute_metrics is called
    THEN per-label metrics are None and support is 0
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
    GIVEN invalid tensor shapes
    WHEN compute_metrics is called
    THEN a ValueError is raised
    """
    logits = torch.tensor([1.0, 2.0])
    targets = torch.tensor([1.0, 0.0])
    masks = torch.tensor([1.0, 1.0])

    with pytest.raises(ValueError):
        compute_metrics(logits=logits, targets=targets, masks=masks)
