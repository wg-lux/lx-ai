# lx_ai/ai_model_data_loader/data_loader_for_model_training.py
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
    TypeGuard,
    cast,
)

from pydantic import ConfigDict, Field, field_validator, model_validator

from lx_dtypes.models.base_models.base_model import AppBaseModel

# -----------------------------------------------------------------------------
# Types: dataset kind dispatch (same idea as old AIDataSet constants)
# -----------------------------------------------------------------------------
DatasetType = Literal["image"]
AIModelType = Literal["image_multilabel_classification"]

# -----------------------------------------------------------------------------
# Minimal dict-shapes for “future DB models”
# (total=False => keys may be absent; avoids TypedDict “None not assignable” errors)
# -----------------------------------------------------------------------------
class FrameDict(TypedDict, total=False):
    id: int
    file_path: str
    old_examination_id: Optional[int]


class LabelSetDict(TypedDict, total=False):
    id: int
    name: str
    version: int
    labels: List["LabelDict"]


class LabelDict(TypedDict, total=False):
    id: int
    name: str
    label_sets: List[LabelSetDict]          # optional metadata for inference
    labelset_versions: List[int]            # optional metadata for inference


class AnnotationDict(TypedDict, total=False):
    frame: FrameDict
    label: LabelDict
    value: bool


# -----------------------------------------------------------------------------
# Output contract: must match what trainer expects
# (This matches the old Django builder output keys exactly.)
# -----------------------------------------------------------------------------
class ImageMultilabelDatasetDataDict(TypedDict):
    image_paths: List[str]
    label_vectors: List[List[Optional[int]]]
    label_masks: List[List[int]]
    labels: List[Any]
    labelset: Any
    frame_ids: List[int]
    old_examination_ids: List[Optional[int]]


class ImageMultilabelDataset(AppBaseModel):
    """
    Validated in-memory representation of an image multi-label training dataset.

    Same contract as old Django version:
      - aligned lists by sample index
      - label_vectors entries are 0/1/None
      - label_masks entries are 0/1
    """

    model_config = AppBaseModel.model_config | ConfigDict(extra="forbid")

    # IMPORTANT: use List[Path] here so callers/builders pass Paths (type-safe).
    # We stringify only when dumping (to_ddict) for metadata/trainer compatibility.
    image_paths: List[Path] = Field(..., min_length=1)

    label_vectors: List[List[Optional[int]]] = Field(..., min_length=1)
    label_masks: List[List[int]] = Field(..., min_length=1)

    labels: List[Any] = Field(default_factory=list)
    labelset: Any = Field(...)

    frame_ids: List[int] = Field(..., min_length=1)
    old_examination_ids: List[Optional[int]] = Field(..., min_length=1)

    # ----------------------------
    # Validators
    # ----------------------------
    @field_validator("image_paths", mode="before")
    @classmethod
    def _coerce_image_paths(cls, v: object) -> List[Path]:
        """
        Accept list[str|Path] and return list[Path].
        Note: this makes runtime robust, but builders should still prefer Paths.
        """
        if not isinstance(v, list):
            raise TypeError("image_paths must be a list")

        out: List[Path] = []
        for item in cast(List[object], v):
            if isinstance(item, Path):
                out.append(item.expanduser())
            elif isinstance(item, str):
                out.append(Path(item).expanduser())
            else:
                raise TypeError(f"image_paths items must be str|Path, got {type(item)!r}")
        return out

    @model_validator(mode="after")
    def _validate_alignment(self) -> "ImageMultilabelDataset":
        n = len(self.image_paths)

        if len(self.label_vectors) != n or len(self.label_masks) != n:
            raise ValueError(
                "image_paths, label_vectors, label_masks must have the same length. "
                f"Got image_paths={n}, label_vectors={len(self.label_vectors)}, label_masks={len(self.label_masks)}"
            )

        if len(self.frame_ids) != n or len(self.old_examination_ids) != n:
            raise ValueError(
                "frame_ids and old_examination_ids must align with samples. "
                f"Got image_paths={n}, frame_ids={len(self.frame_ids)}, old_examination_ids={len(self.old_examination_ids)}"
            )

        c = len(self.label_vectors[0])
        if c == 0:
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

    # ----------------------------
    # ddict export (trainer compatibility)
    # ----------------------------
    @property
    def ddict(self) -> type[ImageMultilabelDatasetDataDict]:
        return ImageMultilabelDatasetDataDict

    def to_ddict(self) -> ImageMultilabelDatasetDataDict:
        return self.ddict(
            image_paths=[str(p) for p in self.image_paths],
            label_vectors=self.label_vectors,
            label_masks=self.label_masks,
            labels=self.labels,
            labelset=self.labelset,
            frame_ids=self.frame_ids,
            old_examination_ids=self.old_examination_ids,
        )


# -----------------------------------------------------------------------------
# Small helpers: dict-or-object access (Pylance-safe)
# -----------------------------------------------------------------------------
def _is_mapping(obj: Any) -> TypeGuard[Mapping[str, Any]]:
    return isinstance(obj, Mapping)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if _is_mapping(obj):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _require(obj: Any, key: str) -> Any:
    v = _get(obj, key, None)
    if v is None:
        raise ValueError(f"Required field '{key}' missing on {type(obj)!r}")
    return v


def _label_key(label: Any) -> int:
    """
    Stable key for a label:
      - if label.id exists and is int -> use it
      - else fallback to built-in id(label)
    """
    lbl_id = _get(label, "id", None)
    if isinstance(lbl_id, int):
        return lbl_id
    return id(label)


# -----------------------------------------------------------------------------
# LabelSet inference (lx-ai replacement for old DB intersection logic)
# -----------------------------------------------------------------------------
def _infer_labelset_from_annotations(annotations: Sequence[Any]) -> LabelSetDict:
    """
    Old Django logic:
      - collect labels from annotations
      - for each label, collect its label_sets ids
      - intersection across all labels
      - require exactly one common labelset id

    New lx-ai logic:
      - works with dicts OR objects
      - expects each label to provide label_sets metadata if inference is requested
    """
    labels: List[Any] = []
    for ann in annotations:
        lbl = _get(ann, "label", None)
        if lbl is not None:
            labels.append(lbl)

    if not labels:
        raise ValueError("Cannot infer LabelSet: annotations contain no labels.")

    labelset_ids_per_label: List[set[int]] = []

    for lbl in labels:
        label_sets_raw = _get(lbl, "label_sets", None)
        if not isinstance(label_sets_raw, list) or not label_sets_raw:
            raise ValueError(
                f"Cannot infer LabelSet: label '{_get(lbl,'name','<unnamed>')}' has no label_sets metadata. "
                "Provide labelset explicitly."
            )

        ids: set[int] = set()
        for ls in cast(List[Any], label_sets_raw):
            ls_id = _get(ls, "id", None)
            if isinstance(ls_id, int):
                ids.add(ls_id)

        if not ids:
            raise ValueError(
                f"Cannot infer LabelSet: label '{_get(lbl,'name','<unnamed>')}' has label_sets but no ids. "
                "Provide labelset explicitly."
            )

        labelset_ids_per_label.append(ids)

    # Manual typed intersection (avoids set.intersection(*list) Unknown issues)
    common: set[int] = set(labelset_ids_per_label[0])
    for s in labelset_ids_per_label[1:]:
        common.intersection_update(s)

    if not common:
        raise ValueError("No common LabelSet across all labels. Provide labelset explicitly.")
    if len(common) > 1:
        raise ValueError(f"More than one common LabelSet found: {sorted(common)}. Provide labelset explicitly.")

    only_id = next(iter(common))
    # IMPORTANT: do NOT set name/version to None; omit keys instead (TypedDict safe)
    return {"id": only_id}


def _labels_in_order_from_labelset(labelset: Any, annotations: Sequence[Any]) -> List[Any]:
    """
    Determine ordered label list (column order).

    Priority (same spirit as old):
      1) labelset.labels if provided (explicit order)
      2) else: stable order from first-seen labels in annotations
    """
    ls_labels_any = _get(labelset, "labels", None)
    if isinstance(ls_labels_any, list) and ls_labels_any:
        return cast(List[Any], ls_labels_any)

    ordered: List[Any] = []
    seen: set[int] = set()

    for ann in annotations:
        lbl = _get(ann, "label", None)
        if lbl is None:
            continue
        k = _label_key(lbl)
        if k not in seen:
            seen.add(k)
            ordered.append(lbl)

    if not ordered:
        raise ValueError("Could not determine labels order: no labels found.")
    return ordered


# -----------------------------------------------------------------------------
# Main builder: lx-ai replacement for build_image_multilabel_dataset_from_db()
# -----------------------------------------------------------------------------
def build_image_multilabel_dataset(
    *,
    dataset_uuid: str,
    annotations: Sequence[Any],
    labelset: Optional[Any] = None,
) -> ImageMultilabelDatasetDataDict:
    """
    Build an in-memory multilabel dataset for image classification.

    Input:
      - annotations: list of annotation-like items.
          each annotation needs:
            - frame: must provide id + file_path (+ optional old_examination_id)
            - label: must provide id/name (and label_sets if labelset inference is needed)
            - value: bool

    Output:
      - keys identical to old Django builder:
          image_paths, label_vectors, label_masks, labels, labelset,
          frame_ids, old_examination_ids
    """
    if not annotations:
        raise ValueError(f"dataset_uuid={dataset_uuid!r} has no annotations.")

    # Decide labelset (explicit or inferred)
    if labelset is None:
        labelset = _infer_labelset_from_annotations(annotations)

    labels_in_order: List[Any] = _labels_in_order_from_labelset(labelset, annotations)
    num_labels = len(labels_in_order)
    if num_labels == 0:
        raise ValueError("LabelSet has no labels (num_labels == 0).")

    # Build label -> column index mapping
    label_index: Dict[int, int] = {}
    for idx, lbl in enumerate(labels_in_order):
        label_index[_label_key(lbl)] = idx

    # Group annotations by frame.id (stable first-seen frame ordering)
    anns_by_frame: Dict[int, List[Any]] = defaultdict(list)
    frames_order: List[int] = []

    for ann in annotations:
        frame = _get(ann, "frame", None)
        frame_id_any = _get(frame, "id", None)
        if not isinstance(frame_id_any, int):
            raise ValueError("Each annotation must have frame.id as int.")
        frame_id = frame_id_any

        if frame_id not in anns_by_frame:
            frames_order.append(frame_id)
        anns_by_frame[frame_id].append(ann)

    # Build dataset arrays (NOTE: image_paths are Paths here, not str => type-safe)
    image_paths: List[Path] = []
    label_vectors: List[List[Optional[int]]] = []
    label_masks: List[List[int]] = []
    frame_ids: List[int] = []
    old_examination_ids: List[Optional[int]] = []

    # Optional cache: frame_by_id
    frame_by_id: Dict[int, Any] = {}

    for frame_id in frames_order:
        frame_annotations = anns_by_frame[frame_id]

        frame = frame_by_id.get(frame_id)
        if frame is None:
            frame = _get(frame_annotations[0], "frame")
            frame_by_id[frame_id] = frame

        frame_ids.append(frame_id)
        old_examination_ids.append(cast(Optional[int], _get(frame, "old_examination_id", None)))

        file_path_raw = _require(frame, "file_path")
        file_path = Path(str(file_path_raw)).expanduser().resolve()
        image_paths.append(file_path)

        vec: List[Optional[int]] = [None] * num_labels

        for ann in frame_annotations:
            lbl = _get(ann, "label", None)
            if lbl is None:
                continue

            idx = label_index.get(_label_key(lbl))
            if idx is None:
                # Label not part of chosen labelset order
                continue

            value_raw = _require(ann, "value")
            value = bool(value_raw)
            vec[idx] = 1 if value else 0

        mask: List[int] = [0 if v is None else 1 for v in vec]

        label_vectors.append(vec)
        label_masks.append(mask)

    ds = ImageMultilabelDataset(
        image_paths=image_paths,                 # Paths -> validated and type-safe
        label_vectors=label_vectors,
        label_masks=label_masks,
        labels=labels_in_order,
        labelset=labelset,
        frame_ids=frame_ids,
        old_examination_ids=old_examination_ids,
    )
    return ds.to_ddict()


# -----------------------------------------------------------------------------
# Dispatch entry point: lx-ai replacement for old build_dataset_for_training()
# -----------------------------------------------------------------------------
class DatasetRequest(TypedDict, total=False):
    dataset_type: DatasetType
    ai_model_type: AIModelType
    dataset_uuid: str
    annotations: List[Any]
    labelset: Any


def build_dataset_for_training(req: DatasetRequest) -> ImageMultilabelDatasetDataDict:
    """
    High-level entry point like old build_dataset_for_training(dataset).

    Required in req:
      - dataset_type: "image"
      - ai_model_type: "image_multilabel_classification"
      - dataset_uuid: str
      - annotations: list

    Optional:
      - labelset: Any
    """
    dataset_type = req.get("dataset_type")
    ai_model_type = req.get("ai_model_type")

    dataset_uuid_any = req.get("dataset_uuid")
    annotations_any = req.get("annotations")

    if not isinstance(dataset_uuid_any, str) or not dataset_uuid_any:
        raise ValueError("req['dataset_uuid'] must be a non-empty str.")
    if not isinstance(annotations_any, list):
        raise ValueError("req['annotations'] must be a list.")

    dataset_uuid: str = dataset_uuid_any
    annotations: List[Any] = annotations_any  # ✅ no cast needed
    labelset = req.get("labelset")

    if dataset_type == "image" and ai_model_type == "image_multilabel_classification":
        return build_image_multilabel_dataset(
            dataset_uuid=dataset_uuid,
            annotations=annotations,
            labelset=labelset,
        )

    raise NotImplementedError(
        f"No dataset builder implemented for dataset_type={dataset_type!r}, ai_model_type={ai_model_type!r}"
    )
