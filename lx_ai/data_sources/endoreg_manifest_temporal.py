from __future__ import annotations

from typing import Any

from lx_ai.data_sources.endoreg_manifest import (
    EndoRegTrainingDatasetManifest,
    EndoRegTrainingSample,
)
from lx_ai.utils.temporal_frame_sampler import (
    sample_annotations_temporally,
)


# -----------------------------------------------------------------------------
# Purpose
# -----------------------------------------------------------------------------
#
# Apply LX-AI's EXISTING temporal frame-sampling policy to EndoRegDB manifest
# samples without changing temporal_frame_sampler.py.
#
# IMPORTANT:
#
# - no PostgreSQL access
# - no EndoRegDB ORM access
# - no Django ORM access
# - no frame materialization
# - no image-path resolution
# - no label-vector mutation
# - no bucket allocation
# - no train/val/test splitting
# - no trainer changes
#
# The existing temporal sampler only needs:
#
#     video_id
#     frame_id
#     frame_number
#     positive label IDs
#
# Therefore we construct a minimal annotation-shaped compatibility view,
# execute the existing sampler unchanged, and map the selected frames back
# to the original manifest samples.
#
# This keeps one authoritative temporal-sampling implementation inside LX-AI.
# -----------------------------------------------------------------------------


def _require_temporal_identity(
    sample: EndoRegTrainingSample,
) -> tuple[int, int, int]:
    """
    Return the identity fields required by the existing temporal sampler.

    Temporal sampling operates on:
        (video_id, frame_id, frame_number)

    These fields are required only when temporal sampling is actually active.
    """

    if sample.video_id is None:
        raise ValueError(
            f"sample={sample.sample_index}: "
            "video_id is required for temporal sampling"
        )

    if sample.frame_id is None:
        raise ValueError(
            f"sample={sample.sample_index}: "
            "frame_id is required for temporal sampling"
        )

    if sample.frame_number is None:
        raise ValueError(
            f"sample={sample.sample_index}: "
            "frame_number is required for temporal sampling"
        )

    return (
        sample.video_id,
        sample.frame_id,
        sample.frame_number,
    )


def _positive_label_ids(
    *,
    manifest: EndoRegTrainingDatasetManifest,
    sample: EndoRegTrainingSample,
) -> list[int]:
    """
    Return positive label IDs for one manifest sample.

    This deliberately follows the semantic information used by the existing
    temporal sampler:

        positive iff mask == 1 and value == 1.0

    Unknown values (mask=0) are not positive.

    Known negatives (mask=1, value=0.0) are not positive.

    This means treat_unlabeled_as_negative does not falsely create positive
    labels and therefore does not alter temporal rare-label scoring.
    """

    positive_ids: list[int] = []

    for label_index, (
        value,
        mask,
    ) in enumerate(
        zip(
            sample.labels,
            sample.label_mask,
        )
    ):
        if mask != 1:
            continue

        if value != 1.0:
            continue

        positive_ids.append(
            manifest.labels[
                label_index
            ].id
        )

    return positive_ids


def _manifest_sample_to_temporal_annotations(
    *,
    manifest: EndoRegTrainingDatasetManifest,
    sample: EndoRegTrainingSample,
) -> list[dict[str, Any]]:
    """
    Build the smallest annotation-shaped representation required by
    sample_annotations_temporally().

    We create one synthetic annotation record per POSITIVE label.

    If a frame has no positive labels, we still create one sentinel record.
    This is essential because the existing temporal sampler must still see
    negative-only / no-positive frames when determining temporal sequences.

    The sentinel has:
        label=None
        value=False

    Therefore it contributes no positive label ID but keeps the frame present.
    """

    (
        video_id,
        frame_id,
        frame_number,
    ) = _require_temporal_identity(
        sample
    )

    frame = {
        "id": frame_id,
        "video_id": video_id,
        "frame_number": frame_number,
    }

    positive_label_ids = (
        _positive_label_ids(
            manifest=manifest,
            sample=sample,
        )
    )

    if not positive_label_ids:
        return [
            {
                "frame": frame,
                "frame_id": frame_id,
                "video_id": video_id,
                "frame_number": frame_number,
                "label": None,
                "value": False,
            }
        ]

    return [
        {
            "frame": frame,
            "frame_id": frame_id,
            "video_id": video_id,
            "frame_number": frame_number,
            "label": {
                "id": label_id,
            },
            "value": True,
        }
        for label_id in positive_label_ids
    ]


def _manifest_to_temporal_annotations(
    manifest: EndoRegTrainingDatasetManifest,
) -> list[dict[str, Any]]:
    """
    Convert all manifest samples into the compatibility representation used
    only for temporal-frame selection.
    """

    annotations: list[
        dict[str, Any]
    ] = []

    for sample in manifest.samples:
        annotations.extend(
            _manifest_sample_to_temporal_annotations(
                manifest=manifest,
                sample=sample,
            )
        )

    return annotations


def _selected_frame_keys(
    sampled_annotations: list[
        dict[str, Any]
    ],
) -> set[tuple[int, int]]:
    """
    Extract selected (video_id, frame_id) pairs from the existing sampler's
    output.

    Using both fields is safer than relying on frame_id alone and keeps the
    compatibility boundary explicit.
    """

    selected: set[
        tuple[int, int]
    ] = set()

    for annotation in sampled_annotations:
        frame = (
            annotation.get("frame")
            or {}
        )

        if not isinstance(
            frame,
            dict,
        ):
            raise ValueError(
                "Temporal sampler returned an "
                "invalid frame payload"
            )

        video_id = (
            frame.get("video_id")
            or annotation.get(
                "video_id"
            )
        )

        frame_id = (
            frame.get("id")
            or frame.get("frame_id")
            or annotation.get(
                "frame_id"
            )
        )

        if (
            video_id is None
            or frame_id is None
        ):
            raise ValueError(
                "Temporal sampler returned a "
                "record without video_id/frame_id"
            )

        selected.add(
            (
                int(video_id),
                int(frame_id),
            )
        )

    return selected


def sample_manifest_samples_temporally(
    manifest: EndoRegTrainingDatasetManifest,
    *,
    enabled: bool,
    max_frames_per_sequence: int,
    min_distance_frames: int,
    sequence_gap_frames: int,
    split_on_label_change: bool,
    rare_label_aware: bool,
) -> list[EndoRegTrainingSample]:
    """
    Apply LX-AI's existing temporal sampling algorithm to EndoReg manifest
    samples.

    The original manifest objects are preserved.

    Only whole samples/frames are removed.

    No label vectors or masks are reconstructed or modified.

    Ordering of retained samples remains the original EndoReg manifest order.

    When sampling is disabled, behavior matches the existing annotation
    sampler and the complete sample list is returned unchanged.
    """

    if not enabled:
        return list(
            manifest.samples
        )

    if max_frames_per_sequence <= 0:
        return list(
            manifest.samples
        )

    compatibility_annotations = (
        _manifest_to_temporal_annotations(
            manifest
        )
    )

    if not compatibility_annotations:
        return list(
            manifest.samples
        )

    sampled_annotations = (
        sample_annotations_temporally(
            compatibility_annotations,
            enabled=enabled,
            max_frames_per_sequence=(
                max_frames_per_sequence
            ),
            min_distance_frames=(
                min_distance_frames
            ),
            sequence_gap_frames=(
                sequence_gap_frames
            ),
            split_on_label_change=(
                split_on_label_change
            ),
            rare_label_aware=(
                rare_label_aware
            ),
        )
    )

    selected_keys = (
        _selected_frame_keys(
            sampled_annotations
        )
    )

    sampled_samples: list[
        EndoRegTrainingSample
    ] = []

    for sample in manifest.samples:
        (
            video_id,
            frame_id,
            _frame_number,
        ) = _require_temporal_identity(
            sample
        )

        if (
            video_id,
            frame_id,
        ) in selected_keys:
            sampled_samples.append(
                sample
            )

    return sampled_samples


__all__ = [
    "sample_manifest_samples_temporally",
]