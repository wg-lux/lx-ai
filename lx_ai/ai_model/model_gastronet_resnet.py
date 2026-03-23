# lx_ai/ai_model/model_gastronet_resnet.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, TypedDict

import torch
from pydantic import ConfigDict, Field, field_validator, model_validator
from torch import nn

from lx_dtypes.models.base_models.base_model import AppBaseModel

from lx_ai.ai_model.model_backbones import MultiLabelBackboneHead, create_multilabel_model


# -----------------------------------------------------------------------------
# Pydantic spec (boundary contract)
# -----------------------------------------------------------------------------
class GastroNetResNetSpecDataDict(TypedDict):
    num_labels: int
    backbone_checkpoint: Optional[str]
    freeze_backbone: bool


class GastroNetResNet50Spec(AppBaseModel):
    """
    Validated config/spec for GastroNetResNet50MultiLabel.

    This keeps your new architecture style:
      - strict schema (extra=forbid)
      - validated invariants
      - stable serialization for metadata (to_ddict)
    """

    model_config = AppBaseModel.model_config | ConfigDict(extra="forbid")

    num_labels: int = Field(..., ge=1)
    backbone_checkpoint: Optional[Path] = None
    freeze_backbone: bool = Field(default=True)

    @field_validator("backbone_checkpoint", mode="before")
    @classmethod
    def _coerce_checkpoint(cls, v: object) -> Optional[Path]:
        if v is None or v == "":
            return None
        if isinstance(v, Path):
            return v.expanduser()
        if isinstance(v, str):
            return Path(v).expanduser()
        raise TypeError(f"backbone_checkpoint must be Path|str|None, got {type(v)!r}")

    @model_validator(mode="after")
    def _normalize_checkpoint(self) -> "GastroNetResNet50Spec":
        if self.backbone_checkpoint is not None:
            # resolve for stable behavior/logging; existence check is handled in backbones loader
            self.backbone_checkpoint = self.backbone_checkpoint.expanduser().resolve()
        return self

    @property
    def ddict(self) -> type[GastroNetResNetSpecDataDict]:
        return GastroNetResNetSpecDataDict

    def to_ddict(self) -> GastroNetResNetSpecDataDict:
        return self.ddict(
            num_labels=self.num_labels,
            backbone_checkpoint=str(self.backbone_checkpoint) if self.backbone_checkpoint else None,
            freeze_backbone=self.freeze_backbone,
        )


# -----------------------------------------------------------------------------
# Model wrapper (keeps old API shape, uses new factory internally)
# -----------------------------------------------------------------------------
class GastroNetResNet50MultiLabel(nn.Module):
    """
    Backwards-compatible wrapper around create_multilabel_model().

    Old behavior:
      - backbone: ResNet50 feature extractor (no ImageNet weights by default)
      - optional checkpoint load (GastroNet)
      - linear head for multi-label logits

    New behavior:
      - delegates backbone construction + checkpoint loading to:
          create_multilabel_model(backbone_name="gastro_rn50", ...)
      - exposes .backbone and .classifier like old code (trainer expects this)
    """

    def __init__(
        self,
        num_labels: int,
        backbone_checkpoint: Optional[Path] = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]

        spec = GastroNetResNet50Spec(
            num_labels=num_labels,
            backbone_checkpoint=backbone_checkpoint,
            freeze_backbone=freeze_backbone,
        )

        model: MultiLabelBackboneHead = create_multilabel_model(
            backbone_name="gastro_rn50",
            num_labels=spec.num_labels,
            backbone_checkpoint=spec.backbone_checkpoint,
            freeze_backbone=spec.freeze_backbone,
        )

        # Preserve the old public module attributes
        self.backbone: nn.Module = model.backbone
        self.classifier: nn.Linear = model.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: torch.Tensor = self.backbone(x)
        if feats.ndim == 4:
            feats = feats.flatten(1)
        return self.classifier(feats)
