# lx_ai/ai_model_dataset/dataset.py
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, List, Optional, Tuple, TypedDict, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from pydantic import ConfigDict, Field, field_validator, model_validator
from torch.utils.data import Dataset
from lx_dtypes.models.base_models.base_model import AppBaseModel


# -----------------------------------------------------------------------------
# 1) Pydantic "spec" model: validate all dataset inputs BEFORE torch sees them
# -----------------------------------------------------------------------------
class MultiLabelDatasetSpecDataDict(TypedDict):
    image_paths: List[str]
    label_vectors: List[List[Optional[int]]]
    label_masks: List[List[int]]
    image_size: int


class MultiLabelDatasetSpec(AppBaseModel):
    """
    Validated in-memory dataset specification.

    This is the "boundary contract" for training data:
    - strict schema
    - strong invariants
    - great error messages
    """

    model_config = AppBaseModel.model_config | ConfigDict(extra="forbid")

    image_paths: List[Path] = Field(..., min_length=1)
    label_vectors: List[List[Optional[int]]] = Field(..., min_length=1)
    label_masks: List[List[int]] = Field(..., min_length=1)
    image_size: int = Field(default=224, ge=16)

    # ----------------------------
    # Converters
    # ----------------------------
    @field_validator("image_paths", mode="before")
    @classmethod
    def _coerce_image_paths(cls, v: object) -> List[Path]:
        """
        Accept list[str|Path] and convert to list[Path].
        """
        if v is None:
            raise TypeError("image_paths must not be None")

        if not isinstance(v, list):
            raise TypeError("image_paths must be a list")

        out: List[Path] = []
        for item in cast(List[object], v):
            if isinstance(item, Path):
                out.append(item.expanduser())
            elif isinstance(item, str):
                out.append(Path(item).expanduser())
            else:
                raise TypeError(
                    f"image_paths items must be str|Path, got {type(item)!r}"
                )
        return out

    @field_validator("label_vectors")
    @classmethod
    def _validate_label_vectors_values(
        cls, v: List[List[Optional[int]]]
    ) -> List[List[Optional[int]]]:
        """
        Enforce values are only {0, 1, None}.
        """
        for i, vec in enumerate(v):
            for j, x in enumerate(vec):
                if x is None:
                    continue
                if x not in (0, 1):
                    raise ValueError(
                        f"label_vectors[{i}][{j}] must be 0|1|None, got {x!r}"
                    )
        return v

    @field_validator("label_masks")
    @classmethod
    def _validate_label_masks_values(cls, v: List[List[int]]) -> List[List[int]]:
        """
        Enforce mask values are only {0, 1}.
        """
        for i, mask in enumerate(v):
            for j, x in enumerate(mask):
                if x not in (0, 1):
                    raise ValueError(f"label_masks[{i}][{j}] must be 0|1, got {x!r}")
        return v

    # ----------------------------
    # Cross-field invariants
    # ----------------------------
    @model_validator(mode="after")
    def _validate_shapes(self) -> "MultiLabelDatasetSpec":
        n_img = len(self.image_paths)
        n_vec = len(self.label_vectors)
        n_msk = len(self.label_masks)

        if not (n_img == n_vec == n_msk):
            raise ValueError(
                "image_paths, label_vectors, label_masks must have the same length. "
                f"Got image_paths={n_img}, label_vectors={n_vec}, label_masks={n_msk}"
            )

        c = len(self.label_vectors[0])
        if c == 0:
            raise ValueError("label_vectors must have at least one label (C>0)")

        for i, (vec, mask) in enumerate(zip(self.label_vectors, self.label_masks)):
            if len(vec) != c:
                raise ValueError(
                    f"label_vectors[{i}] length mismatch: expected {c}, got {len(vec)}"
                )
            if len(mask) != c:
                raise ValueError(
                    f"label_masks[{i}] length mismatch: expected {c}, got {len(mask)}"
                )

            # enforce the semantic consistency: None <-> mask==0
            for j, (x, m) in enumerate(zip(vec, mask)):
                if x is None and m != 0:
                    raise ValueError(
                        f"Inconsistent unknown label at [{i}][{j}]: value=None but mask={m}. "
                        "Expected mask=0 for unknown."
                    )
                if x is not None and m != 1:
                    raise ValueError(
                        f"Inconsistent known label at [{i}][{j}]: value={x} but mask={m}. "
                        "Expected mask=1 for known."
                    )

        return self

    # ----------------------------
    # ddict export
    # ----------------------------
    @property
    def ddict(self) -> type[MultiLabelDatasetSpecDataDict]:
        return MultiLabelDatasetSpecDataDict

    def to_ddict(self) -> MultiLabelDatasetSpecDataDict:
        return self.ddict(
            image_paths=[str(p) for p in self.image_paths],
            label_vectors=self.label_vectors,
            label_masks=self.label_masks,
            image_size=self.image_size,
        )


# -----------------------------------------------------------------------------
# 2) Torch Dataset: consumes validated spec and does fast tensor work
# -----------------------------------------------------------------------------
TorchSample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class EndoMultiLabelDataset(Dataset[TorchSample]):
    """
    PyTorch Dataset wrapping a validated MultiLabelDatasetSpec.
    """

    MEAN: ClassVar[torch.Tensor] = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD: ClassVar[torch.Tensor] = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, spec: MultiLabelDatasetSpec) -> None:
        super().__init__()
        self.spec = spec

        self.image_paths: List[str] = [str(p) for p in spec.image_paths]

        label_vec_list: List[List[int]] = []
        mask_list: List[List[int]] = []
        for vec, mask in zip(spec.label_vectors, spec.label_masks):
            label_vec_list.append([0 if x is None else int(x) for x in vec])
            mask_list.append([int(m) for m in mask])

        self.labels: torch.Tensor = torch.tensor(label_vec_list, dtype=torch.float32)
        self.masks: torch.Tensor = torch.tensor(mask_list, dtype=torch.float32)

        self.num_labels: int = int(self.labels.shape[1])
        self.image_size: int = spec.image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size))

        # Strong typing for numpy -> makes Pylance happy

        arr: NDArray[np.float32] = np.asarray(img, dtype=np.float32) / np.float32(255.0)
        tensor: torch.Tensor = torch.from_numpy(arr)  # type: ignore[arg-type]
        tensor = tensor.permute(2, 0, 1).contiguous()

        tensor = (tensor - self.MEAN) / self.STD
        return tensor

    def __getitem__(self, idx: int) -> TorchSample:
        path = self.image_paths[idx]
        x = self._load_image(path)
        y = self.labels[idx]
        m = self.masks[idx]
        return x, y, m
