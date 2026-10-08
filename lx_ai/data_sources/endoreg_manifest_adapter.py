from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from lx_ai.data_sources.endoreg_manifest import (
    EndoRegTrainingDatasetManifest,
    EndoRegTrainingSample,
)


# -----------------------------------------------------------------------------
# Purpose
# -----------------------------------------------------------------------------
#
# Convert a validated EndoRegDB training manifest into the BASE dataset shape
# already consumed by LX-AI.
#
# This module deliberately DOES NOT perform:
#
# - PostgreSQL access
# - EndoRegDB ORM access
# - Django ORM access
# - temporal sampling
# - zero-positive-label filtering
# - bucket allocation
# - train/validation/test split generation
# - model training
#
# Those operations must remain at their existing logical stages until the
# manifest path has been proven equivalent to the current PostgreSQL path.
#
# Image/frame resolution is injected through ImagePathResolver because the
# existing LX-AI PyTorch dataset ultimately consumes readable local files.
#
# In production, the resolver will later be connected to the appropriate
# EndoReg frame streaming/materialization boundary. This adapter itself must
# remain independent of protected-media paths, encryption keys, and EndoRegDB
# storage internals.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Output metadata types
# -----------------------------------------------------------------------------


class LabelInfo(TypedDict):
    id: int
    name: str


class LabelSetInfo(TypedDict, total=False):
    id: int
    name: str
    version: int
    labels: list[LabelInfo]


class ManifestDatasetDataDict(TypedDict):
    """
    Base dataset output produced from one EndoReg training manifest.

    This intentionally corresponds to the base fields used by the existing
    LX-AI dataset pipeline before bucket/split metadata is attached.
    """

    image_paths: list[str]
    label_vectors: list[list[Optional[int]]]
    label_masks: list[list[int]]

    labels: list[LabelInfo]
    labelset: LabelSetInfo

    frame_ids: list[int]
    old_examination_ids: list[Optional[int]]

    dataset_ids_per_frame: list[int]
    video_ids: list[int]
    annotators_per_frame: list[list[Any]]

    annotation_positive_count: int
    annotation_negative_count: int


# -----------------------------------------------------------------------------
# Image resolution boundary
# -----------------------------------------------------------------------------


ImagePathResolver = Callable[
    [EndoRegTrainingSample],
    Path,
]


# -----------------------------------------------------------------------------
# Label conversion
# -----------------------------------------------------------------------------


def _convert_label_value(
    *,
    value: float,
    mask: int,
    sample_index: int,
    label_index: int,
) -> Optional[int]:
    """
    Convert an EndoReg manifest label/mask pair into the representation used
    by the existing LX-AI training pipeline.

    Current LX-AI semantics:

        value=None, mask=0
            unknown / unlabeled

        value=0, mask=1
            known negative

        value=1, mask=1
            known positive

    EndoReg's manifest stores numeric values even for masked/unknown labels,
    therefore mask=0 must become value=None inside LX-AI.
    """

    if mask == 0:
        return None

    if mask != 1:
        raise ValueError(
            f"sample={sample_index}, label={label_index}: "
            f"label_mask must be 0 or 1, got {mask!r}"
        )

    if value == 0.0:
        return 0

    if value == 1.0:
        return 1

    raise ValueError(
        f"sample={sample_index}, label={label_index}: "
        "current LX-AI multilabel training requires binary "
        f"label values 0.0 or 1.0 when mask=1, got {value!r}"
    )


# -----------------------------------------------------------------------------
# Label / LabelSet metadata
# -----------------------------------------------------------------------------


def _build_label_info(
    manifest: EndoRegTrainingDatasetManifest,
) -> list[LabelInfo]:
    """
    Preserve EndoRegDB's manifest label ordering exactly.

    IMPORTANT:
    Do not remove zero-positive labels here.

    The current PostgreSQL path performs temporal sampling before the existing
    dataset builder performs zero-positive-label filtering. Filtering here
    would happen too early and could change training behavior.
    """

    return [
        {
            "id": label.id,
            "name": label.name,
        }
        for label in manifest.labels
    ]


def _manifest_labelset_id(
    manifest: EndoRegTrainingDatasetManifest,
) -> int | None:
    """
    Read the LabelSet database ID supplied by EndoRegDB provenance.

    The current EndoRegDB manifest builder writes:

        provenance["labelset_id"]

    It is metadata rather than a top-level manifest field.
    """

    raw_value = manifest.provenance.get(
        "labelset_id"
    )

    if raw_value is None:
        return None

    # bool is a subclass of int in Python. Do not accept True/False as IDs.
    if not isinstance(raw_value, int) or isinstance(raw_value, bool):
        raise ValueError(
            "manifest.provenance['labelset_id'] "
            "must be an integer when provided"
        )

    if raw_value < 1:
        raise ValueError(
            "manifest.provenance['labelset_id'] "
            "must be greater than or equal to 1"
        )

    return raw_value


def _build_labelset_info(
    manifest: EndoRegTrainingDatasetManifest,
    labels: list[LabelInfo],
) -> LabelSetInfo:
    """
    Reconstruct the labelset-shaped metadata expected by the existing LX-AI
    trainer.

    Current EndoRegDB manifest behavior:

    - labelset name is attached to every AITrainingLabel
    - labelset version is attached to every AITrainingLabel
    - labelset ID is stored in manifest.provenance["labelset_id"]

    All labels in one manifest are expected to belong to the same LabelSet.
    """

    first_label = manifest.labels[0]

    labelset_name = first_label.labelset_name
    labelset_version = first_label.labelset_version

    for label in manifest.labels:
        if label.labelset_name != labelset_name:
            raise ValueError(
                "EndoReg training manifest contains labels from "
                "different labelset names"
            )

        if label.labelset_version != labelset_version:
            raise ValueError(
                "EndoReg training manifest contains labels from "
                "different labelset versions"
            )

    result: LabelSetInfo = {
        "labels": labels,
    }

    labelset_id = _manifest_labelset_id(
        manifest
    )

    if labelset_id is not None:
        result["id"] = labelset_id

    if labelset_name is not None:
        result["name"] = labelset_name

    if labelset_version is not None:
        result["version"] = labelset_version

    return result


# -----------------------------------------------------------------------------
# Legacy metadata compatibility
# -----------------------------------------------------------------------------


def _get_old_examination_id(
    sample: EndoRegTrainingSample,
) -> Optional[int]:
    """
    Preserve old_examination_id only when EndoReg explicitly supplies it.

    old_examination_id is legacy compatibility metadata and is not part of the
    modern required training-manifest sample contract.

    The active bucket allocator groups primarily by video_id. Existing
    examination checks remain available as secondary legacy leakage checks.
    """

    value = sample.metadata.get(
        "old_examination_id"
    )

    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"sample={sample.sample_index}: "
            "metadata['old_examination_id'] "
            "must be an integer or null"
        )

    return value


def _get_annotators(
    sample: EndoRegTrainingSample,
) -> list[Any]:
    """
    Preserve annotators only if the manifest explicitly supplies annotators.

    IMPORTANT:
    EndoRegDB currently emits information_source_names.

    An InformationSource is not automatically equivalent to the old SQL
    'annotator' field, so we intentionally do NOT map:

        information_source_names -> annotators_per_frame

    Doing so would invent semantics.

    If a future EndoReg manifest explicitly provides metadata["annotators"],
    this compatibility field can preserve it.
    """

    value = sample.metadata.get(
        "annotators"
    )

    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(
            f"sample={sample.sample_index}: "
            "metadata['annotators'] must be a list when provided"
        )

    return list(value)


# -----------------------------------------------------------------------------
# Annotation-count diagnostics
# -----------------------------------------------------------------------------


def _annotation_ids_by_label(
    sample: EndoRegTrainingSample,
) -> dict[str, list[int]]:
    """
    Read explicit frame-annotation provenance produced by EndoRegDB.

    Current EndoRegDB manifest samples contain:

        metadata["annotation_ids_by_label"] = {
            "<label-name>": [annotation_id, ...],
            ...
        }

    This is useful because the legacy PostgreSQL loader's
    annotation_positive_count / annotation_negative_count values count
    annotation rows, not merely frame/label matrix cells.
    """

    raw_value = sample.metadata.get(
        "annotation_ids_by_label"
    )

    if raw_value is None:
        return {}

    if not isinstance(raw_value, dict):
        raise ValueError(
            f"sample={sample.sample_index}: "
            "metadata['annotation_ids_by_label'] must be a mapping"
        )

    result: dict[str, list[int]] = {}

    for raw_label_name, raw_ids in raw_value.items():
        label_name = str(
            raw_label_name
        )

        if not isinstance(raw_ids, list):
            raise ValueError(
                f"sample={sample.sample_index}: "
                "metadata['annotation_ids_by_label'] values "
                "must be lists"
            )

        annotation_ids: list[int] = []

        for raw_id in raw_ids:
            if not isinstance(raw_id, int) or isinstance(raw_id, bool):
                raise ValueError(
                    f"sample={sample.sample_index}: "
                    "annotation IDs must be integers"
                )

            if raw_id < 1:
                raise ValueError(
                    f"sample={sample.sample_index}: "
                    "annotation IDs must be greater than or equal to 1"
                )

            annotation_ids.append(
                raw_id
            )

        result[label_name] = (
            annotation_ids
        )

    return result


def _count_explicit_frame_annotations(
    *,
    sample: EndoRegTrainingSample,
    manifest: EndoRegTrainingDatasetManifest,
) -> tuple[int, int]:
    """
    Count positive/negative explicit frame annotations when EndoRegDB provides
    annotation provenance.

    This is intentionally based on annotation IDs rather than on every known
    matrix cell.

    Why:
    The current PostgreSQL path calculates its diagnostics from raw annotation
    rows:

        one DB annotation row -> one positive or negative count

    Counting every known manifest cell would change that meaning,
    particularly when treat_unlabeled_as_negative=True.

    Segment-derived labels are not counted here as explicit frame annotation
    rows. They still remain fully represented in label_vectors/label_masks.
    """

    annotation_ids = (
        _annotation_ids_by_label(
            sample
        )
    )

    if not annotation_ids:
        return 0, 0

    label_index_by_name = {
        label.name: label.index
        for label in manifest.labels
    }

    positive_count = 0
    negative_count = 0

    for (
        label_name,
        ids,
    ) in annotation_ids.items():
        label_index = (
            label_index_by_name.get(
                label_name
            )
        )

        if label_index is None:
            raise ValueError(
                f"sample={sample.sample_index}: "
                "metadata['annotation_ids_by_label'] "
                f"references unknown label {label_name!r}"
            )

        value = sample.labels[
            label_index
        ]

        mask = sample.label_mask[
            label_index
        ]

        if mask != 1:
            raise ValueError(
                f"sample={sample.sample_index}, "
                f"label={label_name!r}: "
                "explicit annotation provenance exists "
                "but label_mask is not 1"
            )

        if value == 1.0:
            positive_count += len(
                ids
            )
        elif value == 0.0:
            negative_count += len(
                ids
            )
        else:
            raise ValueError(
                f"sample={sample.sample_index}, "
                f"label={label_name!r}: "
                "explicit annotation provenance requires "
                f"a binary value, got {value!r}"
            )

    return (
        positive_count,
        negative_count,
    )


# -----------------------------------------------------------------------------
# Main adapter
# -----------------------------------------------------------------------------


def build_lx_ai_dataset_from_endoreg_manifest(
    *,
    manifest: EndoRegTrainingDatasetManifest,
    resolve_image_path: ImagePathResolver,
) -> ManifestDatasetDataDict:
    """
    Convert one validated EndoRegDB training manifest into LX-AI's BASE
    in-memory dataset representation.

    This conversion preserves:

    - manifest sample order
    - manifest label order
    - 0 / 1 / None label semantics
    - label masks
    - frame IDs
    - video IDs
    - dataset IDs
    - available legacy metadata
    - available explicit annotation-count diagnostics

    This function deliberately stops BEFORE:

    - temporal sampling
    - zero-positive-label filtering
    - bucket allocation
    - split generation
    - training

    Those operations must be applied later in the same logical order used by
    the current PostgreSQL pipeline.
    """

    if manifest.dataset_id is None:
        raise ValueError(
            "EndoReg training manifest must contain dataset_id "
            "for current LX-AI training compatibility"
        )

    labels = _build_label_info(
        manifest
    )

    labelset = _build_labelset_info(
        manifest,
        labels,
    )

    image_paths: list[str] = []

    label_vectors: list[
        list[Optional[int]]
    ] = []

    label_masks: list[
        list[int]
    ] = []

    frame_ids: list[int] = []

    old_examination_ids: list[
        Optional[int]
    ] = []

    dataset_ids_per_frame: list[
        int
    ] = []

    video_ids: list[int] = []

    annotators_per_frame: list[
        list[Any]
    ] = []

    annotation_positive_count = 0
    annotation_negative_count = 0

    for sample in manifest.samples:
        # -----------------------------------------------------------------
        # Required identity fields
        # -----------------------------------------------------------------

        if sample.frame_id is None:
            raise ValueError(
                f"sample={sample.sample_index}: "
                "frame_id is required by the current LX-AI "
                "training pipeline"
            )

        if sample.video_id is None:
            raise ValueError(
                f"sample={sample.sample_index}: "
                "video_id is required by the current LX-AI "
                "training pipeline"
            )

        # -----------------------------------------------------------------
        # Resolve actual readable image
        # -----------------------------------------------------------------
        #
        # The current PyTorch dataset ultimately uses PIL.Image.open(path).
        #
        # The adapter itself must not know:
        # - PROTECTED_MEDIA_ROOT
        # - LX_RUNTIME_ROOT
        # - LX_ANNOTATE_MASTER_KEY_FILE
        # - encrypted storage implementation
        # - EndoRegDB ORM
        #
        # Those concerns belong behind the resolver/provider boundary.
        # -----------------------------------------------------------------

        resolved_path = (
            resolve_image_path(
                sample
            )
        )

        if not isinstance(
            resolved_path,
            Path,
        ):
            raise TypeError(
                "resolve_image_path() "
                "must return pathlib.Path"
            )

        resolved_path = (
            resolved_path
            .expanduser()
            .resolve()
        )

        if not resolved_path.is_file():
            raise FileNotFoundError(
                "Resolved training image does not exist: "
                f"{resolved_path}. "
                f"sample_index={sample.sample_index}, "
                f"frame_id={sample.frame_id}, "
                f"video_id={sample.video_id}, "
                f"frame_number={sample.frame_number}"
            )

        image_paths.append(
            str(resolved_path)
        )

        # -----------------------------------------------------------------
        # Preserve label/mask semantics
        # -----------------------------------------------------------------

        vector: list[
            Optional[int]
        ] = []

        mask_row: list[int] = []

        for (
            label_index,
            (
                value,
                mask,
            ),
        ) in enumerate(
            zip(
                sample.labels,
                sample.label_mask,
            )
        ):
            converted = (
                _convert_label_value(
                    value=value,
                    mask=mask,
                    sample_index=(
                        sample.sample_index
                    ),
                    label_index=(
                        label_index
                    ),
                )
            )

            vector.append(
                converted
            )

            mask_row.append(
                int(mask)
            )

        label_vectors.append(
            vector
        )

        label_masks.append(
            mask_row
        )

        # -----------------------------------------------------------------
        # Preserve aligned sample metadata
        # -----------------------------------------------------------------

        frame_ids.append(
            sample.frame_id
        )

        old_examination_ids.append(
            _get_old_examination_id(
                sample
            )
        )

        dataset_ids_per_frame.append(
            manifest.dataset_id
        )

        video_ids.append(
            sample.video_id
        )

        annotators_per_frame.append(
            _get_annotators(
                sample
            )
        )

        # -----------------------------------------------------------------
        # Preserve legacy annotation-count meaning as closely as the manifest
        # contract allows.
        # -----------------------------------------------------------------

        (
            sample_positive_count,
            sample_negative_count,
        ) = _count_explicit_frame_annotations(
            sample=sample,
            manifest=manifest,
        )

        annotation_positive_count += (
            sample_positive_count
        )

        annotation_negative_count += (
            sample_negative_count
        )

    # ---------------------------------------------------------------------
    # Defensive alignment validation
    # ---------------------------------------------------------------------

    sample_count = len(
        manifest.samples
    )

    lengths = {
        "image_paths": len(
            image_paths
        ),
        "label_vectors": len(
            label_vectors
        ),
        "label_masks": len(
            label_masks
        ),
        "frame_ids": len(
            frame_ids
        ),
        "old_examination_ids": len(
            old_examination_ids
        ),
        "dataset_ids_per_frame": len(
            dataset_ids_per_frame
        ),
        "video_ids": len(
            video_ids
        ),
        "annotators_per_frame": len(
            annotators_per_frame
        ),
    }

    invalid_lengths = {
        name: length
        for name, length
        in lengths.items()
        if length != sample_count
    }

    if invalid_lengths:
        raise ValueError(
            "Manifest adapter produced "
            "misaligned dataset arrays. "
            f"Expected each length={sample_count}; "
            f"actual={invalid_lengths}"
        )

    # ---------------------------------------------------------------------
    # Defensive label-column validation
    # ---------------------------------------------------------------------

    label_count = len(
        labels
    )

    for sample_index, (
        vector,
        mask,
    ) in enumerate(
        zip(
            label_vectors,
            label_masks,
        )
    ):
        if len(vector) != label_count:
            raise ValueError(
                f"sample={sample_index}: "
                "adapter label vector length "
                f"{len(vector)} does not match "
                f"label count {label_count}"
            )

        if len(mask) != label_count:
            raise ValueError(
                f"sample={sample_index}: "
                "adapter label mask length "
                f"{len(mask)} does not match "
                f"label count {label_count}"
            )

    return {
        "image_paths": image_paths,
        "label_vectors": label_vectors,
        "label_masks": label_masks,
        "labels": labels,
        "labelset": labelset,
        "frame_ids": frame_ids,
        "old_examination_ids": (
            old_examination_ids
        ),
        "dataset_ids_per_frame": (
            dataset_ids_per_frame
        ),
        "video_ids": video_ids,
        "annotators_per_frame": (
            annotators_per_frame
        ),
        "annotation_positive_count": (
            annotation_positive_count
        ),
        "annotation_negative_count": (
            annotation_negative_count
        ),
    }


__all__ = [
    "ImagePathResolver",
    "LabelInfo",
    "LabelSetInfo",
    "ManifestDatasetDataDict",
    "build_lx_ai_dataset_from_endoreg_manifest",
]