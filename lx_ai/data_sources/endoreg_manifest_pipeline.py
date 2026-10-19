from __future__ import annotations

from pathlib import Path
from typing import Optional

from lx_ai.data_sources.endoreg_manifest import (
    EndoRegTrainingDatasetManifest,
)
from lx_ai.data_sources.endoreg_manifest_adapter import (
    ImagePathResolver,
    LabelInfo,
    LabelSetInfo,
    ManifestDatasetDataDict,
    build_lx_ai_dataset_from_endoreg_manifest,
)
from lx_ai.data_sources.endoreg_manifest_temporal import (
    sample_manifest_samples_temporally,
)


# -----------------------------------------------------------------------------
# Purpose
# -----------------------------------------------------------------------------
#
# Compose the EndoRegDB manifest input path into the BASE dataset shape expected
# by LX-AI.
#
# The order here deliberately mirrors the relevant ordering of the existing
# PostgreSQL path:
#
#     source data
#         ↓
#     temporal sampling
#         ↓
#     dataset construction
#         ↓
#     drop labels with zero positive samples
#
# This module deliberately DOES NOT perform:
#
# - PostgreSQL access
# - EndoRegDB ORM access
# - Django ORM access
# - bucket allocation
# - train/validation/test split generation
# - trainer invocation
# - model construction
#
# Those remain in their current LX-AI locations.
#
# This file is the composition boundary between:
#
#     EndoReg semantic training manifest
#
# and
#
#     existing LX-AI base training data
#
# -----------------------------------------------------------------------------


def _label_has_positive_sample(
    *,
    label_index: int,
    label_vectors: list[list[Optional[int]]],
    label_masks: list[list[int]],
) -> bool:
    """
    Return True if a label has at least one known positive sample.

    Equivalent semantic requirement to the existing
    build_image_multilabel_dataset() zero-positive-label filter.

    Unknown labels do not count as positive.

        value=None, mask=0 -> unknown
        value=0,    mask=1 -> known negative
        value=1,    mask=1 -> known positive
    """

    for vector, mask in zip(
        label_vectors,
        label_masks,
    ):
        if (
            mask[label_index] == 1
            and vector[label_index] == 1
        ):
            return True

    return False


def _positive_label_indices(
    data: ManifestDatasetDataDict,
) -> list[int]:
    """
    Determine which label columns survive the existing LX-AI rule:

        drop labels with zero positive samples

    This must happen AFTER temporal sampling because temporal sampling can
    remove every positive frame for a label.
    """

    label_count = len(
        data["labels"]
    )

    kept_indices = [
        label_index
        for label_index
        in range(label_count)
        if _label_has_positive_sample(
            label_index=label_index,
            label_vectors=(
                data["label_vectors"]
            ),
            label_masks=(
                data["label_masks"]
            ),
        )
    ]

    if not kept_indices:
        raise ValueError(
            "No labels with positive samples remain after filtering."
        )

    return kept_indices


def _filter_label_columns(
    rows: list[list[Optional[int]]],
    kept_indices: list[int],
) -> list[list[Optional[int]]]:
    return [
        [
            row[index]
            for index in kept_indices
        ]
        for row in rows
    ]


def _filter_mask_columns(
    rows: list[list[int]],
    kept_indices: list[int],
) -> list[list[int]]:
    return [
        [
            row[index]
            for index in kept_indices
        ]
        for row in rows
    ]


def _filter_labels(
    labels: list[LabelInfo],
    kept_indices: list[int],
) -> list[LabelInfo]:
    return [
        labels[index]
        for index in kept_indices
    ]


def _filter_labelset(
    *,
    labelset: LabelSetInfo,
    filtered_labels: list[LabelInfo],
) -> LabelSetInfo:
    """
    Preserve LabelSet identity metadata while replacing its labels with the
    filtered training label space.

    The existing PostgreSQL builder also returns the selected labelset together
    with the labels that survive dataset construction.
    """

    result: LabelSetInfo = {}

    labelset_id = labelset.get(
        "id"
    )

    if labelset_id is not None:
        result["id"] = labelset_id

    labelset_name = labelset.get(
        "name"
    )

    if labelset_name is not None:
        result["name"] = labelset_name

    labelset_version = labelset.get(
        "version"
    )

    if labelset_version is not None:
        result["version"] = (
            labelset_version
        )

    result["labels"] = (
        filtered_labels
    )

    return result


def _drop_zero_positive_labels(
    data: ManifestDatasetDataDict,
) -> ManifestDatasetDataDict:
    """
    Apply the existing LX-AI zero-positive-label behavior after temporal
    sampling.

    IMPORTANT:
    annotation_positive_count / annotation_negative_count are intentionally
    left untouched.

    The existing PostgreSQL pipeline computes those annotation diagnostics
    before build_image_multilabel_dataset() performs zero-positive-label
    filtering.

    Preserving the counts here therefore matches the current ordering.
    """

    kept_indices = (
        _positive_label_indices(
            data
        )
    )

    # Fast path: no label was removed.
    if len(kept_indices) == len(
        data["labels"]
    ):
        return data

    filtered_labels = (
        _filter_labels(
            data["labels"],
            kept_indices,
        )
    )

    filtered_labelset = (
        _filter_labelset(
            labelset=data["labelset"],
            filtered_labels=(
                filtered_labels
            ),
        )
    )

    return {
        "image_paths": list(
            data["image_paths"]
        ),
        "label_vectors": (
            _filter_label_columns(
                data["label_vectors"],
                kept_indices,
            )
        ),
        "label_masks": (
            _filter_mask_columns(
                data["label_masks"],
                kept_indices,
            )
        ),
        "labels": filtered_labels,
        "labelset": filtered_labelset,
        "frame_ids": list(
            data["frame_ids"]
        ),
        "old_examination_ids": list(
            data["old_examination_ids"]
        ),
        "dataset_ids_per_frame": list(
            data["dataset_ids_per_frame"]
        ),
        "video_ids": list(
            data["video_ids"]
        ),
        "annotators_per_frame": [
            list(annotators)
            for annotators
            in data[
                "annotators_per_frame"
            ]
        ],
        "annotation_positive_count": (
            data[
                "annotation_positive_count"
            ]
        ),
        "annotation_negative_count": (
            data[
                "annotation_negative_count"
            ]
        ),
    }


def build_lx_ai_dataset_from_endoreg_manifest_pipeline(
    *,
    manifest: EndoRegTrainingDatasetManifest,
    resolve_image_path: ImagePathResolver,
    temporal_sampling_enabled: bool,
    temporal_max_frames_per_sequence: int,
    temporal_min_distance_frames: int,
    temporal_sequence_gap_frames: int,
    temporal_split_on_label_change: bool,
    temporal_rare_label_aware: bool,
) -> ManifestDatasetDataDict:
    """
    Build the LX-AI base training dataset from an EndoReg training manifest.

    Processing order:

        1. validate manifest
           - already completed before this function is called

        2. apply EXISTING LX-AI temporal frame sampling

        3. preserve only the selected original manifest samples

        4. convert selected manifest samples into LX-AI base dataset arrays

        5. drop labels with zero positive samples

    The result intentionally stops BEFORE:

        - video bucket allocation
        - split generation
        - split integrity verification
        - trainer invocation

    Those remain owned by the existing central LX-AI pipeline.
    """

    sampled_samples = (
        sample_manifest_samples_temporally(
            manifest,
            enabled=(
                temporal_sampling_enabled
            ),
            max_frames_per_sequence=(
                temporal_max_frames_per_sequence
            ),
            min_distance_frames=(
                temporal_min_distance_frames
            ),
            sequence_gap_frames=(
                temporal_sequence_gap_frames
            ),
            split_on_label_change=(
                temporal_split_on_label_change
            ),
            rare_label_aware=(
                temporal_rare_label_aware
            ),
        )
    )

    if not sampled_samples:
        raise ValueError(
            "EndoReg training manifest contains no samples "
            "after temporal sampling."
        )

    # -----------------------------------------------------------------
    # Preserve the validated manifest and replace only the sample list.
    #
    # model_copy() avoids converting through JSON/dicts and preserves the
    # validated Pydantic sample objects.
    # -----------------------------------------------------------------

    sampled_manifest = (
        manifest.model_copy(
            update={
                "samples": sampled_samples,
            }
        )
    )

    base_data = (
        build_lx_ai_dataset_from_endoreg_manifest(
            manifest=sampled_manifest,
            resolve_image_path=(
                resolve_image_path
            ),
        )
    )

    return _drop_zero_positive_labels(
        base_data
    )


__all__ = [
    "build_lx_ai_dataset_from_endoreg_manifest_pipeline",
]