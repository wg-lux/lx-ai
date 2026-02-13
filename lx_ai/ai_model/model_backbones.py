#lx_ai/ai_model/model_backbones.py
from __future__ import annotations

from pathlib import Path
from typing import Dict,  Literal, Optional, TypedDict, cast

import torch
from torch import nn
from pydantic import ConfigDict, Field, field_validator, model_validator

from lx_dtypes.models.base_models.base_model import AppBaseModel

from collections.abc import Mapping as AbcMapping

from typing import Any, Dict, TypeGuard
from lx_ai.utils.logging_utils import subsection

# -----------------------------------------------------------------------------
# Public types (keep trainer/config compatibility)
# -----------------------------------------------------------------------------
BackboneName = Literal[
    "gastro_rn50",
    "resnet50_imagenet",
    "resnet50_random",
    "efficientnet_b0_imagenet",
]


class BackboneConfigDataDict(TypedDict):
    backbone_name: BackboneName
    num_labels: int
    backbone_checkpoint: Optional[str]
    freeze_backbone: bool


class BackboneBuildResult(TypedDict):
    backbone: nn.Module
    in_features: int


# -----------------------------------------------------------------------------
# Pydantic: validated factory input (boundary contract)
# -----------------------------------------------------------------------------
class MultiLabelModelFactorySpec(AppBaseModel):
    model_config = AppBaseModel.model_config | ConfigDict(extra="forbid")

    backbone_name: BackboneName = Field(default="gastro_rn50")
    num_labels: int = Field(..., ge=1)
    backbone_checkpoint: Optional[Path] = None
    freeze_backbone: bool = True

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
    def _normalize_checkpoint(self) -> "MultiLabelModelFactorySpec":
        if self.backbone_checkpoint is not None:
            self.backbone_checkpoint = self.backbone_checkpoint.expanduser().resolve()
        return self

    @property
    def ddict(self) -> type[BackboneConfigDataDict]:
        return BackboneConfigDataDict

    def to_ddict(self) -> BackboneConfigDataDict:
        return self.ddict(
            backbone_name=self.backbone_name,
            num_labels=self.num_labels,
            backbone_checkpoint=str(self.backbone_checkpoint) if self.backbone_checkpoint else None,
            freeze_backbone=self.freeze_backbone,
        )


# -----------------------------------------------------------------------------
# Model: backbone + linear head (trainer expects .backbone and .classifier)
# -----------------------------------------------------------------------------
class MultiLabelBackboneHead(nn.Module):
    """
    Contract used by trainer:
      - model.backbone.parameters()
      - model.classifier.parameters()
      - forward(x) -> logits [B, C]
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        num_labels: int,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__() ## pyright: ignore[reportUnknownMemberType] 
        # TODO pyright issue: torch is installed but pyright cannot find its types so have to resolve and remove the error ignore 
        self.backbone: nn.Module = backbone
        self.classifier: nn.Linear = nn.Linear(in_features, num_labels)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.ndim == 4:
            feats = feats.flatten(1)
        return self.classifier(feats)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _require_torchvision() -> None:
    """
    Runtime dependency check.
    NOTE: Pylance errors about missing torchvision are handled by `type: ignore`
    on the actual imports below.
    """
    try:
        import torchvision  # type: ignore[import-not-found]  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "torchvision is required for model_backbones (resnet/efficientnet) but it is not available "
            "in the current Python environment.\n\n"
            "Fix:\n"
            "  - Add torchvision to your Nix dev shell / Python env\n"
            "  - Ensure VSCode/Pylance uses the same interpreter as your shell\n"
        ) from e


def _is_mapping(obj: Any) -> TypeGuard[AbcMapping[Any, Any]]:
    """
    True if obj is a Mapping (dict-like).

    TypeGuard is critical: it allows Pylance to narrow types inside `if _is_mapping(...)`.
    """
    return isinstance(obj, AbcMapping)


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Accept common checkpoint formats:
      1) plain state_dict: Mapping[str, Tensor]
      2) wrapper mapping with key 'state_dict' containing the state dict

    Returns:
      dict[str, torch.Tensor] cleaned for torchvision backbones.
    """
    state_obj: Any = obj

    # unwrap {"state_dict": ...} if present
    if _is_mapping(state_obj):
        # after TypeGuard: state_obj is AbcMapping[Any, Any]
        if "state_dict" in state_obj:
            inner_obj: Any = state_obj["state_dict"]
            if _is_mapping(inner_obj):
                state_obj = inner_obj

    if not _is_mapping(state_obj):
        raise TypeError(
            f"Checkpoint must be a mapping/dict-like object, got {type(state_obj)!r}"
        )

    cleaned: Dict[str, torch.Tensor] = {}

    # after TypeGuard: state_obj is AbcMapping[Any, Any] so .items() is valid
    for k_raw, v_raw in state_obj.items():
        if not isinstance(k_raw, str):
            continue
        if not isinstance(v_raw, torch.Tensor):
            continue

        new_k = k_raw
        for prefix in ("module.", "backbone.", "encoder.", "model."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix) :]

        # ResNet final layer is fc.* — drop it for feature extractor usage
        if new_k.startswith("fc."):
            continue

        cleaned[new_k] = v_raw

    return cleaned



def _load_torchvision_resnet50(*, imagenet: bool) -> nn.Module:
    _require_torchvision()
    import torchvision.models as tvm  # type: ignore[import-not-found]

    fn = getattr(tvm, "resnet50")

    # Handle old/new torchvision APIs safely
    try:
        if imagenet:
            weights_enum = getattr(tvm, "ResNet50_Weights", None)
            if weights_enum is not None:
                weights = getattr(weights_enum, "IMAGENET1K_V1", None)
                if weights is not None:
                    return cast(nn.Module, fn(weights=weights))
            return cast(nn.Module, fn(pretrained=True))
        return cast(nn.Module, fn(weights=None))
    except TypeError:
        return cast(nn.Module, fn(pretrained=bool(imagenet)))


def _load_torchvision_efficientnet_b0_imagenet() -> nn.Module:
    _require_torchvision()
    import torchvision.models as tvm  # type: ignore[import-not-found]

    fn = getattr(tvm, "efficientnet_b0")

    try:
        weights_enum = getattr(tvm, "EfficientNet_B0_Weights", None)
        if weights_enum is not None:
            weights = getattr(weights_enum, "IMAGENET1K_V1", None)
            if weights is not None:
                return cast(nn.Module, fn(weights=weights))
        return cast(nn.Module, fn(pretrained=True))
    except TypeError:
        return cast(nn.Module, fn(pretrained=True))


def _build_resnet50_backbone(
    *,
    imagenet: bool,
    checkpoint: Optional[Path],
) -> BackboneBuildResult:
    base = _load_torchvision_resnet50(imagenet=imagenet)

    if checkpoint is not None and checkpoint.is_file():
        loaded_obj: object = torch.load(checkpoint, map_location="cpu")
        state_dict = _extract_state_dict(loaded_obj)

        incompatible = base.load_state_dict(state_dict, strict=False)
        # In PyTorch this object has fields: missing_keys, unexpected_keys
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))

        subsection("BACKBONE INITIALIZATION")
        print(f"  Checkpoint loaded   : {checkpoint}")
        if missing:
            print(f"  Missing keys ignored: {len(missing)}")


    # Feature extractor: remove final fc
    backbone = nn.Sequential(*list(base.children())[:-1])  # [B, 2048, 1, 1]

    fc = getattr(base, "fc", None)
    if not isinstance(fc, nn.Linear):
        raise TypeError("Expected ResNet50 to have attribute .fc as nn.Linear")
    in_features = int(fc.in_features)

    return {"backbone": backbone, "in_features": in_features}


def _build_efficientnet_b0_backbone() -> BackboneBuildResult:
    base = _load_torchvision_efficientnet_b0_imagenet()

    features = getattr(base, "features", None)
    classifier = getattr(base, "classifier", None)

    if not isinstance(features, nn.Module):
        raise TypeError("Expected EfficientNet-B0 to have .features as nn.Module")
    if not isinstance(classifier, nn.Module):
        raise TypeError("Expected EfficientNet-B0 to have .classifier as nn.Module")

    backbone = nn.Sequential(
        features,
        nn.AdaptiveAvgPool2d(1),
    )

    in_features: Optional[int] = None
    if isinstance(classifier, nn.Sequential):
        for m in classifier.modules():
            if isinstance(m, nn.Linear):
                in_features = int(m.in_features)
                break

    if in_features is None:
        raise TypeError("Could not infer in_features from EfficientNet-B0 classifier")

    return {"backbone": backbone, "in_features": in_features}


# -----------------------------------------------------------------------------
# Public API (same signature used by trainer)
# -----------------------------------------------------------------------------
def create_multilabel_model(
    backbone_name: str,
    num_labels: int,
    backbone_checkpoint: Optional[Path],
    freeze_backbone: bool = True,
) -> MultiLabelBackboneHead:
    spec = MultiLabelModelFactorySpec(
        backbone_name=cast(BackboneName, backbone_name.lower()),
        num_labels=num_labels,
        backbone_checkpoint=backbone_checkpoint,
        freeze_backbone=freeze_backbone,
    )

    if spec.backbone_name == "gastro_rn50":
        res = _build_resnet50_backbone(imagenet=False, checkpoint=spec.backbone_checkpoint)
    elif spec.backbone_name == "resnet50_imagenet":
        res = _build_resnet50_backbone(imagenet=True, checkpoint=None)
    elif spec.backbone_name == "resnet50_random":
        res = _build_resnet50_backbone(imagenet=False, checkpoint=None)
    elif spec.backbone_name == "efficientnet_b0_imagenet":
        res = _build_efficientnet_b0_backbone()
    else:
        raise ValueError(f"Unknown backbone_name={spec.backbone_name!r}")

    return MultiLabelBackboneHead(
        backbone=res["backbone"],
        in_features=int(res["in_features"]),
        num_labels=spec.num_labels,
        freeze_backbone=spec.freeze_backbone,
    )
