# lx_ai/utils/data_loader_for_model_input.py
from __future__ import annotations

from lx_ai.utils.db_loader_for_model_input import load_annotations_from_postgres

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING, TypedDict

from pydantic import ConfigDict, Field, field_validator, model_validator
from lx_ai.utils.data_loader_for_model_training import (
    build_image_multilabel_dataset,
)
from lx_ai.ai_model_config.config import TrainingConfig


from lx_ai.utils.db_loader_for_model_input import (
    load_annotations_from_postgres,
    load_labelset_from_postgres,
)


# -----------------------------------------------------------------------------
# IMPORTANT (Pylance fix):
# Your AppBaseModel lives in lx_dtypes and Pylance may treat it as "untyped base".
# We keep runtime using your real AppBaseModel, but for static analysis we provide
# a typed stand-in so fields like List[LabelInfo] don't become list[Unknown].
# -----------------------------------------------------------------------------
from lx_dtypes.models.base_models.base_model import AppBaseModel

"""if TYPE_CHECKING:
    from pydantic import BaseModel as _TypedBaseModel

    class AppBaseModel(_TypedBaseModel):  # type: ignore[misc]
        model_config = ConfigDict(arbitrary_types_allowed=True)
else:
    from lx_dtypes.models.base_models.base_model import AppBaseModel"""


# -----------------------------------------------------------------------------
# Defaults you gave (can be overridden from CLI or function arguments)
# -----------------------------------------------------------------------------
DEFAULT_IMAGE_DIR = Path("/home/admin/dev/legacy_images/images")
DEFAULT_JSONL_PATH = Path("/home/admin/dev/legacy_images/legacy_img_dicts.jsonl")

DEFAULT_LABELS: List[str] = [
    "appendix",
    "blood",
    "ileocaecalvalve",
    "ileum",
    "low_quality",
    "outside",
    "polyp",
    "water_jet",
    "wound",
]


# -----------------------------------------------------------------------------
# Output contract (trainer compatibility)
# -----------------------------------------------------------------------------
class LabelInfo(TypedDict):
    id: int
    name: str


class LabelSetInfo(TypedDict, total=False):
    id: int
    name: str
    version: int
    labels: List[LabelInfo]


class ImageMultilabelDatasetDataDict(TypedDict):
    image_paths: List[str]
    label_vectors: List[List[Optional[int]]]
    label_masks: List[List[int]]
    labels: List[LabelInfo]
    labelset: LabelSetInfo
    frame_ids: List[int]
    old_examination_ids: List[Optional[int]]
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    bucket_policy: Dict[str, Any]
    bucket_sizes: Dict[str, int]
    role_sizes: Dict[str, int]


def _empty_labelset_info() -> LabelSetInfo:
    # Typed default factory for a TypedDict (Pylance-safe)
    return {}


# -----------------------------------------------------------------------------
# JSONL record schema (your legacy input)
# -----------------------------------------------------------------------------
class LegacyJsonlRecord(AppBaseModel):
    """
    One line from legacy_img_dicts.jsonl

    Example:
      {"labels": ["appendix","polyp"], "old_examination_id": 25, "old_id": 479228, "filename": "479228.jpg"}
    """

    model_config = getattr(AppBaseModel, "model_config", ConfigDict()) | ConfigDict(extra="forbid")

    labels: List[str] = Field(default_factory=list)
    old_examination_id: Optional[int] = None
    old_id: int
    filename: str

    @field_validator("filename")
    @classmethod
    def _filename_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("filename must be a non-empty string")
        return s


# -----------------------------------------------------------------------------
# Validated dataset model (internal)
# -----------------------------------------------------------------------------
class ImageMultilabelDataset(AppBaseModel):
    """
    Same structure as old Django output, but validated + JSON-friendly.

    NOTE:
      - image_paths is List[Path] internally (safer),
      - but to_ddict returns List[str] for trainer compatibility.
    """

    model_config = getattr(AppBaseModel, "model_config", ConfigDict()) | ConfigDict(extra="forbid")

    image_paths: List[Path] = Field(..., min_length=1)
    label_vectors: List[List[Optional[int]]] = Field(..., min_length=1)
    label_masks: List[List[int]] = Field(..., min_length=1)

    # ✅ Pylance-safe now (LabelInfo is known + base model is typed for checking)
    labels:List[LabelInfo] = Field(default_factory=list)


    # ✅ TypedDict needs typed default_factory
    labelset: LabelSetInfo = Field(default_factory=_empty_labelset_info)

    frame_ids: List[int] = Field(..., min_length=1)
    old_examination_ids: List[Optional[int]] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_alignment(self) -> "ImageMultilabelDataset":
        n = len(self.image_paths)

        if len(self.label_vectors) != n or len(self.label_masks) != n:
            raise ValueError(
                "image_paths, label_vectors, label_masks must have same length. "
                f"Got image_paths={n}, label_vectors={len(self.label_vectors)}, label_masks={len(self.label_masks)}"
            )

        if len(self.frame_ids) != n or len(self.old_examination_ids) != n:
            raise ValueError(
                "frame_ids and old_examination_ids must align with samples. "
                f"Got frame_ids={len(self.frame_ids)}, old_examination_ids={len(self.old_examination_ids)}, samples={n}"
            )

        c = len(self.label_vectors[0])
        if c <= 0:
            raise ValueError("label_vectors must have at least one label column (C>0)")

        for i, (vec, mask) in enumerate(zip(self.label_vectors, self.label_masks)):
            if len(vec) != c:
                raise ValueError(f"label_vectors[{i}] length mismatch: expected {c}, got {len(vec)}")
            if len(mask) != c:
                raise ValueError(f"label_masks[{i}] length mismatch: expected {c}, got {len(mask)}")

            for j, m in enumerate(mask):
                if m not in (0, 1):
                    raise ValueError(f"label_masks[{i}][{j}] must be 0|1, got {m!r}")

            for j, x in enumerate(vec):
                if x is None:
                    continue
                if x not in (0, 1):
                    raise ValueError(f"label_vectors[{i}][{j}] must be 0|1|None, got {x!r}")

        return self

    def to_ddict(self) -> ImageMultilabelDatasetDataDict:
        return {
            "image_paths": [str(p) for p in self.image_paths],
            "label_vectors": self.label_vectors,
            "label_masks": self.label_masks,
            "labels": self.labels,
            "labelset": self.labelset,
            "frame_ids": self.frame_ids,
            "old_examination_ids": self.old_examination_ids,
        }


# -----------------------------------------------------------------------------
# Core loader helpers
# -----------------------------------------------------------------------------
def _read_jsonl(path: Path) -> Iterable[LegacyJsonlRecord]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {str(path)}: {e}") from e
            yield LegacyJsonlRecord.model_validate(obj)


@dataclass(frozen=True)
class BuildStats:
    total: int
    kept: int
    missing_files: int
    unknown_labels_count: int
    unknown_labels_unique: int


def _build_from_legacy_jsonl(
    *,
    image_dir: Path,
    jsonl_path: Path,
    labels_in_order: Sequence[str],
    assume_missing_is_negative: bool,
    require_existing_files: bool,
) -> tuple[ImageMultilabelDataset, BuildStats]:
    image_dir = image_dir.expanduser().resolve()
    jsonl_path = jsonl_path.expanduser().resolve()

    label_to_idx: Dict[str, int] = {name: i for i, name in enumerate(labels_in_order)}
    labels_info: List[LabelInfo] = [{"id": i, "name": name} for i, name in enumerate(labels_in_order)]

    # Minimal labelset metadata (trainer/debug-friendly)
    labelset_info: LabelSetInfo = {
        "id": 1,
        "name": "legacy_jsonl_labelset",
        "version": 2,
        "labels": labels_info,
    }

    image_paths: List[Path] = []
    label_vectors: List[List[Optional[int]]] = []
    label_masks: List[List[int]] = []
    frame_ids: List[int] = []
    old_examination_ids: List[Optional[int]] = []

    unknown_labels_seen: Dict[str, int] = {}
    missing_files = 0
    total = 0
    kept = 0

    for rec in _read_jsonl(jsonl_path):
        total += 1

        img_path = (image_dir / rec.filename).expanduser().resolve()
        if require_existing_files and not img_path.is_file():
            missing_files += 1
            continue

        # Mode A: missing label => 0, mask=1 (supervised closed world)
        # Mode B: missing label => None, mask=0 (unknown like old DB case)
        if assume_missing_is_negative:
            vec: List[Optional[int]] = [0] * len(labels_in_order)
            mask: List[int] = [1] * len(labels_in_order)
        else:
            vec = [None] * len(labels_in_order)
            mask = [0] * len(labels_in_order)

        for name in rec.labels:
            idx = label_to_idx.get(name)
            if idx is None:
                unknown_labels_seen[name] = unknown_labels_seen.get(name, 0) + 1
                continue
            vec[idx] = 1
            mask[idx] = 1

        image_paths.append(img_path)
        label_vectors.append(vec)
        label_masks.append(mask)
        frame_ids.append(int(rec.old_id))
        old_examination_ids.append(rec.old_examination_id)
        kept += 1

    ds = ImageMultilabelDataset(
        image_paths=image_paths,
        label_vectors=label_vectors,
        label_masks=label_masks,
        labels=labels_info,
        labelset=labelset_info,
        frame_ids=frame_ids,
        old_examination_ids=old_examination_ids,
    )

    stats = BuildStats(
        total=total,
        kept=kept,
        missing_files=missing_files,
        unknown_labels_count=sum(unknown_labels_seen.values()),
        unknown_labels_unique=len(unknown_labels_seen),
    )
    return ds, stats


# -----------------------------------------------------------------------------
# Public API (KEEP NAME STABLE like old project)
# -----------------------------------------------------------------------------
def build_dataset_for_training(
    *,
    config: TrainingConfig,
) -> ImageMultilabelDatasetDataDict:
    """
    Replacement for old Django build_dataset_for_training(dataset, labelset=None)

    In lx-ai we don't have DB models yet:
      - dataset/labelset are accepted for future compatibility but ignored for now
      - data is loaded from JSONL + IMAGE_DIR

    Returns dict with same keys as old system.
    """
    if config.data_source == "jsonl":
        ds_model, _ = _build_from_legacy_jsonl(
            image_dir=DEFAULT_IMAGE_DIR,
            jsonl_path=config.jsonl_path,
            labels_in_order=DEFAULT_LABELS,
            assume_missing_is_negative=config.treat_unlabeled_as_negative,
            require_existing_files=True,
        )
        ds = ds_model.to_ddict()

        from lx_ai.ai_model_split.bucket_splitter import split_indices_by_bucket_policy
        
        train_idx, val_idx, test_idx, bucket_ids, bucket_sizes, role_sizes = split_indices_by_bucket_policy(
            frame_ids=ds["frame_ids"],
            old_examination_ids=ds["old_examination_ids"],
            policy=config.bucket_policy,
        )

        from lx_ai.ai_model_split.bucket_integrity_checker import verify_bucket_integrity

        verify_bucket_integrity(
            frame_ids=ds["frame_ids"],
            old_examination_ids=ds["old_examination_ids"],
            bucket_ids=bucket_ids,
        )

        
        ds["train_indices"] = train_idx
        ds["val_indices"] = val_idx
        ds["test_indices"] = test_idx
        ds["bucket_policy"] = config.bucket_policy.to_meta()
        ds["bucket_sizes"] = bucket_sizes
        ds["role_sizes"] = role_sizes
        
        return ds


    if config.data_source == "postgres":
        all_annotations = []
        '''annotations = load_annotations_from_postgres(
            dataset_id=config.dataset_id
        )'''
        for ds_id in config.dataset_ids:
            anns = load_annotations_from_postgres(dataset_id=ds_id)
            all_annotations.extend(anns)
        
        annotations = all_annotations

        labelset = load_labelset_from_postgres(
            labelset_id=config.labelset_id,
            labelset_version = config.labelset_version_to_train,)
        
        ds = build_image_multilabel_dataset(
        dataset_uuid=config.dataset_uuid,
        annotations=annotations,
        labelset=labelset,
        treat_unlabeled_as_negative=config.treat_unlabeled_as_negative,
    )

        # NEW: compute immutable split indices once, here (dataset building stage)
        from lx_ai.ai_model_split.bucket_splitter import split_indices_by_bucket_policy
        
        train_idx, val_idx, test_idx, bucket_ids, bucket_sizes, role_sizes = split_indices_by_bucket_policy(
            frame_ids=ds["frame_ids"],
            old_examination_ids=ds["old_examination_ids"],
            policy=config.bucket_policy,
        )

        from lx_ai.ai_model_split.bucket_integrity_checker import verify_bucket_integrity

        verify_bucket_integrity(
            frame_ids=ds["frame_ids"],
            old_examination_ids=ds["old_examination_ids"],
            bucket_ids=bucket_ids,
        )

        
        ds["train_indices"] = train_idx
        ds["val_indices"] = val_idx
        ds["test_indices"] = test_idx
        ds["bucket_policy"] = config.bucket_policy.to_meta()
        ds["bucket_sizes"] = bucket_sizes
        ds["role_sizes"] = role_sizes
        
        return ds


    raise ValueError(f"Unknown data_source={config.data_source!r}")


# -----------------------------------------------------------------------------
# CLI entrypoint (runnable script)
# -----------------------------------------------------------------------------
'''def _cli() -> int:
    p = argparse.ArgumentParser(
        prog="data_loader_for_model_input",
        description="Build a multi-label training dataset dict from legacy JSONL + images directory.",
    )
    p.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    p.add_argument("--jsonl-path", type=str, default=str(DEFAULT_JSONL_PATH))

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--assume-missing-is-negative",
        action="store_true",
        help="Missing labels become 0 with mask=1 (default).",
    )
    mode.add_argument(
        "--assume-missing-is-unknown",
        action="store_true",
        help="Missing labels become None with mask=0 (unknown/ignored).",
    )

    p.add_argument(
        "--no-require-existing-files",
        action="store_true",
        help="Do not skip records when image files are missing.",
    )
    

    args = p.parse_args()

    assume_missing_is_negative = True
    if args.assume_missing_is_unknown:
        assume_missing_is_negative = False
    elif args.assume_missing_is_negative:
        assume_missing_is_negative = True

    _ = build_dataset_for_training(
        dataset=None,
        labelset=None,
        image_dir=Path(args.image_dir),
        jsonl_path=Path(args.jsonl_path),
        labels_in_order=tuple(DEFAULT_LABELS),
        assume_missing_is_negative=assume_missing_is_negative,
        require_existing_files=not args.no_require_existing_files,
    )
    return 0'''


'''if __name__ == "__main__":
    raise SystemExit(_cli())
'''