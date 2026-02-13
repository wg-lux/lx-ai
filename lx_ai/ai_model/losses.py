#lx_ai/ai_model/losses.py
from __future__ import annotations

from typing import Optional, TypedDict

import torch
from pydantic import ConfigDict, Field, field_validator

from lx_dtypes.models.base_models.base_model import AppBaseModel

from typing import cast
# -----------------------------------------------------------------------------
# Typed export (“ddict”) – consistent with your TrainingConfig pattern
# -----------------------------------------------------------------------------
class LossConfigDataDict(TypedDict):
    alpha: float
    gamma: float
    eps: float
    use_class_weights: bool


# -----------------------------------------------------------------------------
# Pydantic config model (GOOD place to use AppBaseModel)
# - strict schema
# - validated ranges
# - can be dumped to metadata consistently
# -----------------------------------------------------------------------------
class FocalLossConfig(AppBaseModel):
    """
    Configuration for multi-label focal loss with optional class weights.

    This is intentionally small and stable:
    - It is safe to store in metadata
    - It has strong validation guarantees
    - It does NOT attempt to validate torch tensors (that stays runtime)
    """

    model_config = AppBaseModel.model_config | ConfigDict(extra="forbid")

    alpha: float = Field(default=0.25, ge=0.0, le=1.0)
    gamma: float = Field(default=2.0, ge=0.0)
    eps: float = Field(default=1e-6, gt=0.0)

    # If False, ignore provided class_weights even if passed.
    use_class_weights: bool = Field(default=True)

    @field_validator("eps")
    @classmethod
    def _validate_eps_reasonable(cls, v: float) -> float:
        # Big eps can destroy gradients / probabilities.
        if v >= 1e-1:
            raise ValueError("eps is too large; use values like 1e-6 or 1e-8.")
        return v

    @property #i think we can delete this now
    def ddict(self) -> type[LossConfigDataDict]:
        return LossConfigDataDict
    
    def to_ddict(self) -> LossConfigDataDict:
        d = self.model_dump()
        return {
            "alpha": float(d["alpha"]),
            "gamma": float(d["gamma"]),
            "eps": float(d["eps"]),
            "use_class_weights": bool(d["use_class_weights"]),
        }


# -----------------------------------------------------------------------------
# Internal runtime validators (fast + Pylance-friendly)
# -----------------------------------------------------------------------------
def _require_2d(name: str, x: torch.Tensor) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D [N,C] or [B,C]. Got shape={tuple(x.shape)}")


def _require_same_shape(a_name: str, a: torch.Tensor, b_name: str, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{b_name} must match {a_name} shape. {a_name}={tuple(a.shape)}, {b_name}={tuple(b.shape)}"
        )


def _require_non_empty(name: str, x: torch.Tensor) -> None:
    if x.numel() == 0:
        raise ValueError(f"{name} must not be empty")


def _require_class_weights(weights: torch.Tensor, num_labels: int) -> None:
    if weights.ndim != 1:
        raise ValueError(f"class_weights must be 1D [C]. Got shape={tuple(weights.shape)}")
    if int(weights.shape[0]) != int(num_labels):
        raise ValueError(
            f"class_weights length must match C. Got len={int(weights.shape[0])}, C={int(num_labels)}"
        )
    if torch.any(weights <= 0):
        raise ValueError("class_weights must be > 0")


# -----------------------------------------------------------------------------
# Public API (KEEP EXACT signatures and behavior as old endoreg_db version)
# -----------------------------------------------------------------------------
def compute_class_weights(
    labels: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute per-label weights based on positive counts.

    labels: [N, C] in {0,1} (float/int OK)
    masks:  [N, C] in {0,1} (float/int OK), 1 = known, 0 = unknown

    w_j = 1 / (pos_j + eps), normalized so that mean(w) ≈ 1.
    """
    _require_2d("labels", labels)
    _require_2d("masks", masks)
    _require_same_shape("labels", labels, "masks", masks)
    _require_non_empty("labels", labels)
    _require_non_empty("masks", masks)

    # Keep old semantics: mask > 0.5 counts as known.
    known: torch.Tensor = masks > 0.5
    pos_counts: torch.Tensor = (labels * known).sum(dim=0)  # [C]

    raw_weights: torch.Tensor = 1.0 / (pos_counts + float(eps))
    mean_w: torch.Tensor = raw_weights.mean().clamp(min=float(eps))
    norm_weights: torch.Tensor = raw_weights / mean_w
    return norm_weights  # [C]


def focal_loss_with_mask(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Multi-label focal loss with:
      - optional per-label class weights
      - mask to ignore unknown labels

    logits: [B, C] raw model outputs
    targets: [B, C] in {0,1} (float/int OK)
    masks: [B, C] in {0,1} (float/int OK)
    class_weights: [C] or None
    """
    cfg = FocalLossConfig(alpha=alpha, gamma=gamma, eps=eps)

    _require_2d("logits", logits)
    _require_2d("targets", targets)
    _require_2d("masks", masks)
    _require_same_shape("logits", logits, "targets", targets)
    _require_same_shape("logits", logits, "masks", masks)
    _require_non_empty("logits", logits)

    c = int(logits.shape[1])
    if cfg.use_class_weights and class_weights is not None:
        _require_class_weights(class_weights, num_labels=c)

    prob: torch.Tensor = torch.sigmoid(logits).clamp(cfg.eps, 1.0 - cfg.eps)  # [B, C]

    # p_t: prob if y=1, (1-prob) if y=0
    pt: torch.Tensor = prob * targets + (1.0 - prob) * (1.0 - targets)

    alpha_factor: torch.Tensor = cfg.alpha * targets + (1.0 - cfg.alpha) * (1.0 - targets)
    focal_factor: torch.Tensor = (1.0 - pt) ** cfg.gamma

    loss: torch.Tensor = -alpha_factor * focal_factor * torch.log(pt)  # [B, C]

    if cfg.use_class_weights and class_weights is not None:
        loss = loss * class_weights.view(1, -1)

    # Apply mask → ignore unknown labels
    loss = loss * masks

    denom: torch.Tensor = masks.sum().clamp(min=1.0)
    return loss.sum() / denom
