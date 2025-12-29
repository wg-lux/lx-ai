# lx_ai/ai_model_matrics/metrics.py
from __future__ import annotations

from typing import List, Optional, TypedDict

import torch


# -----------------------------------------------------------------------------
# Typed output shapes (Pylance + documentation)
# -----------------------------------------------------------------------------
class PerLabelMetrics(TypedDict):
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    support: int


class MetricsResult(TypedDict):
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int
    per_label: List[PerLabelMetrics]


# -----------------------------------------------------------------------------
# Helpers (small + explicit)
# -----------------------------------------------------------------------------
_EPS: float = 1e-6


def _require_2d(name: str, x: torch.Tensor) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D [N,C]. Got shape={tuple(x.shape)}")


def _require_same_shape(a_name: str, a: torch.Tensor, b_name: str, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{b_name} must match {a_name} shape. "
            f"{a_name}={tuple(a.shape)}, {b_name}={tuple(b.shape)}"
        )


# -----------------------------------------------------------------------------
# Main API (same meaning as endoreg_db version)
# -----------------------------------------------------------------------------
def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
) -> MetricsResult:
    """
    Computes multi-label metrics with masking (same semantics as your old code).

    Inputs:
      - logits:   [N, C] raw model outputs (float)
      - targets:  [N, C] labels, expected 0/1 (float or int is OK)
      - masks:    [N, C] mask, expected 0/1 (float or int is OK)
      - threshold: sigmoid threshold to convert probs -> predictions

    Output:
      - Global metrics:
          precision, recall, f1, accuracy, tp/fp/tn/fn
      - Per-label metrics:
          list of dicts {precision, recall, f1, support}
        Where support = number of positives among VALID (mask==1) samples.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0,1]. Got {threshold}")

    _require_2d("logits", logits)
    _require_2d("targets", targets)
    _require_2d("masks", masks)
    _require_same_shape("logits", logits, "targets", targets)
    _require_same_shape("logits", logits, "masks", masks)

    # ---- predictions (sigmoid + threshold) ----
    probs: torch.Tensor = torch.sigmoid(logits)
    preds_i: torch.Tensor = (probs >= threshold).to(dtype=torch.int64)

    # ---- enforce integer tensors for metric math ----
    targets_i: torch.Tensor = targets.to(dtype=torch.int64)
    masks_i: torch.Tensor = masks.to(dtype=torch.int64)

    # ---- only evaluate where mask == 1 ----
    """preds_i = preds_i * masks_i
    targets_i = targets_i * masks_i

    # ---- global confusion counts ----
    tp: int = int((preds_i * targets_i).sum().item())
    fp: int = int((preds_i * (1 - targets_i)).sum().item())
    fn: int = int(((1 - preds_i) * targets_i).sum().item())
    tn: int = int(((1 - preds_i) * (1 - targets_i)).sum().item())

"""
        # ---- only evaluate where mask == 1 (TRUE ignore semantics) ----
    valid: torch.Tensor = masks_i == 1

    # Flatten valid entries (micro metrics over all valid label positions)
    p_all: torch.Tensor = preds_i[valid]
    t_all: torch.Tensor = targets_i[valid]

    # ---- global confusion counts over VALID entries only ----
    tp: int = int(((p_all == 1) & (t_all == 1)).sum().item())
    fp: int = int(((p_all == 1) & (t_all == 0)).sum().item())
    fn: int = int(((p_all == 0) & (t_all == 1)).sum().item())
    tn: int = int(((p_all == 0) & (t_all == 0)).sum().item())


    precision: float = float(tp / (tp + fp + _EPS))
    recall: float = float(tp / (tp + fn + _EPS))
    f1: float = float(2 * precision * recall / (precision + recall + _EPS))
    accuracy: float = float((tp + tn) / (tp + tn + fp + fn + _EPS))

    # ---- per-label metrics ----
    per_label: List[PerLabelMetrics] = []
    num_labels: int = int(targets_i.shape[1])

    for j in range(num_labels):
        t_j: torch.Tensor = targets_i[:, j]
        p_j: torch.Tensor = preds_i[:, j]
        m_j: torch.Tensor = masks_i[:, j]

        valid_idx: torch.Tensor = m_j == 1
        valid_count: int = int(valid_idx.sum().item())

        if valid_count == 0:
            per_label.append({"precision": None, "recall": None, "f1": None, "support": 0})
            continue

        t_j = t_j[valid_idx]
        p_j = p_j[valid_idx]

        tp_j: int = int(((p_j == 1) & (t_j == 1)).sum().item())
        fp_j: int = int(((p_j == 1) & (t_j == 0)).sum().item())
        fn_j: int = int(((p_j == 0) & (t_j == 1)).sum().item())

        precision_j: float = float(tp_j / (tp_j + fp_j + _EPS))
        recall_j: float = float(tp_j / (tp_j + fn_j + _EPS))
        f1_j: float = float(2 * precision_j * recall_j / (precision_j + recall_j + _EPS))

        # same meaning as old code:
        # support = number of positives among valid targets for that label
        support: int = int(t_j.sum().item())

        per_label.append(
            {"precision": precision_j, "recall": recall_j, "f1": f1_j, "support": support}
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "per_label": per_label,
    }
