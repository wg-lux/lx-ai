from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lx_ai.data_sources.endoreg_manifest import (
    EndoRegTrainingDatasetManifest,
    validate_endoreg_training_manifest,
)
from lx_ai.data_sources.endoreg_manifest_temporal import (
    sample_manifest_samples_temporally,
)
from lx_ai.utils.temporal_frame_sampler import (
    sample_annotations_temporally,
)


# -----------------------------------------------------------------------------
# Test fixtures / helpers
# -----------------------------------------------------------------------------


def _label(
    *,
    label_id: int,
    name: str,
    index: int,
) -> dict[str, Any]:
    return {
        "id": label_id,
        "name": name,
        "index": index,
        "labelset_name": "test-labelset",
        "labelset_version": 1,
    }


def _sample(
    *,
    sample_index: int,
    frame_id: int,
    video_id: int,
    frame_number: int,
    labels: list[float],
    masks: list[int],
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "path": None,
        "relative_path": None,
        "labels": labels,
        "label_mask": masks,
        "group_id": f"video-{video_id}",
        "frame_id": frame_id,
        "video_id": video_id,
        "video_uuid": f"uuid-{video_id}",
        "frame_number": frame_number,
        "timestamp": float(frame_number),
        "metadata": {},
    }


def _manifest(
    samples: list[dict[str, Any]],
) -> EndoRegTrainingDatasetManifest:
    payload = {
        "schema_version": "1.0",
        "dataset_id": 1,
        "name": "temporal-test",
        "description": None,
        "modality": "frame",
        "task_kind": "multilabel_classification",
        "labels": [
            _label(
                label_id=10,
                name="polyp",
                index=0,
            ),
            _label(
                label_id=20,
                name="blood",
                index=1,
            ),
        ],
        "samples": samples,
        "frame_format": {
            "check_required": False,
            "status": "not_checked",
            "checked_frame_count": 0,
            "expected_image_format": None,
            "expected_width": None,
            "expected_height": None,
            "expected_mode": None,
            "preprocessing_strategy": (
                "preserve_dimensions_black_mask"
            ),
            "recommended_model_input_strategy": (
                "crop_to_endoscope_roi"
            ),
            "crop_templates_by_video_uuid": {},
            "notes": [],
            "errors": [],
        },
        "class_frequencies": [
            0.0,
            0.0,
        ],
        "provenance": {
            "source": "endoreg_db.AIDataSet",
            "dataset_id": 1,
            "labelset_id": 5,
            "labelset_name": "test-labelset",
            "labelset_version": 1,
        },
    }

    return validate_endoreg_training_manifest(
        payload
    )


def _annotation(
    *,
    frame_id: int,
    video_id: int,
    frame_number: int,
    label_id: int | None,
    value: bool,
) -> dict[str, Any]:
    return {
        "frame": {
            "id": frame_id,
            "video_id": video_id,
            "frame_number": frame_number,
        },
        "frame_id": frame_id,
        "video_id": video_id,
        "frame_number": frame_number,
        "label": (
            {
                "id": label_id,
            }
            if label_id is not None
            else None
        ),
        "value": value,
    }


def _frame_ids_from_annotations(
    annotations: list[dict[str, Any]],
) -> list[int]:
    frame_ids: set[int] = set()

    for annotation in annotations:
        frame = (
            annotation.get("frame")
            or {}
        )

        frame_id = (
            frame.get("id")
            or annotation.get("frame_id")
        )

        assert frame_id is not None

        frame_ids.add(
            int(frame_id)
        )

    return sorted(
        frame_ids
    )


def _frame_ids_from_manifest_samples(
    samples: list[Any],
) -> list[int]:
    return sorted(
        int(sample.frame_id)
        for sample in samples
        if sample.frame_id is not None
    )


# -----------------------------------------------------------------------------
# Basic behavior
# -----------------------------------------------------------------------------


def test_manifest_temporal_disabled_preserves_all_samples() -> None:
    manifest = _manifest(
        [
            _sample(
                sample_index=0,
                frame_id=100,
                video_id=1,
                frame_number=10,
                labels=[1.0, 0.0],
                masks=[1, 0],
            ),
            _sample(
                sample_index=1,
                frame_id=101,
                video_id=1,
                frame_number=20,
                labels=[1.0, 0.0],
                masks=[1, 0],
            ),
        ]
    )

    sampled = (
        sample_manifest_samples_temporally(
            manifest,
            enabled=False,
            max_frames_per_sequence=1,
            min_distance_frames=1,
            sequence_gap_frames=100,
            split_on_label_change=True,
            rare_label_aware=True,
        )
    )

    assert [
        sample.frame_id
        for sample in sampled
    ] == [
        100,
        101,
    ]


def test_manifest_temporal_preserves_original_sample_objects() -> None:
    manifest = _manifest(
        [
            _sample(
                sample_index=0,
                frame_id=100,
                video_id=1,
                frame_number=10,
                labels=[1.0, 0.0],
                masks=[1, 0],
            ),
            _sample(
                sample_index=1,
                frame_id=101,
                video_id=1,
                frame_number=20,
                labels=[1.0, 0.0],
                masks=[1, 0],
            ),
        ]
    )

    sampled = (
        sample_manifest_samples_temporally(
            manifest,
            enabled=True,
            max_frames_per_sequence=1,
            min_distance_frames=1,
            sequence_gap_frames=100,
            split_on_label_change=False,
            rare_label_aware=False,
        )
    )

    assert len(sampled) == 1

    selected = sampled[0]

    assert any(
        selected is original
        for original in manifest.samples
    )


# -----------------------------------------------------------------------------
# Positive-label semantics
# -----------------------------------------------------------------------------


def test_unknown_label_is_not_temporal_positive() -> None:
    """
    mask=0 means unknown.

    Even though the manifest stores numeric 0.0 in that column, it must not
    become a temporal positive label.
    """

    manifest = _manifest(
        [
            _sample(
                sample_index=0,
                frame_id=100,
                video_id=1,
                frame_number=10,
                labels=[
                    1.0,
                    0.0,
                ],
                masks=[
                    1,
                    0,
                ],
            )
        ]
    )

    sampled = (
        sample_manifest_samples_temporally(
            manifest,
            enabled=True,
            max_frames_per_sequence=1,
            min_distance_frames=1,
            sequence_gap_frames=100,
            split_on_label_change=True,
            rare_label_aware=True,
        )
    )

    assert _frame_ids_from_manifest_samples(
        sampled
    ) == [
        100
    ]


def test_negative_only_frame_is_not_dropped() -> None:
    """
    A frame with no positive labels must still participate in temporal
    sequencing.

    This verifies the sentinel compatibility record.
    """

    manifest = _manifest(
        [
            _sample(
                sample_index=0,
                frame_id=100,
                video_id=1,
                frame_number=10,
                labels=[
                    0.0,
                    0.0,
                ],
                masks=[
                    1,
                    1,
                ],
            )
        ]
    )

    sampled = (
        sample_manifest_samples_temporally(
            manifest,
            enabled=True,
            max_frames_per_sequence=10,
            min_distance_frames=1,
            sequence_gap_frames=100,
            split_on_label_change=True,
            rare_label_aware=True,
        )
    )

    assert _frame_ids_from_manifest_samples(
        sampled
    ) == [
        100
    ]


# -----------------------------------------------------------------------------
# Exact parity with existing annotation sampler
# -----------------------------------------------------------------------------


def test_manifest_temporal_matches_existing_sampler_for_single_label_sequence() -> None:
    """
    Both representations describe the same frames:

        frame 100 -> polyp positive
        frame 101 -> polyp positive
        frame 102 -> polyp positive

    The same existing temporal algorithm must select the same frame IDs.
    """

    old_annotations = [
        _annotation(
            frame_id=100,
            video_id=1,
            frame_number=10,
            label_id=10,
            value=True,
        ),
        _annotation(
            frame_id=101,
            video_id=1,
            frame_number=20,
            label_id=10,
            value=True,
        ),
        _annotation(
            frame_id=102,
            video_id=1,
            frame_number=30,
            label_id=10,
            value=True,
        ),
    ]

    manifest = _manifest(
        [
            _sample(
                sample_index=0,
                frame_id=100,
                video_id=1,
                frame_number=10,
                labels=[1.0, 0.0],
                masks=[1, 0],
            ),
            _sample(
                sample_index=1,
                frame_id=101,
                video_id=1,
                frame_number=20,
                labels=[1.0, 0.0],
                masks=[1, 0],
            ),
            _sample(
                sample_index=2,
                frame_id=102,
                video_id=1,
                frame_number=30,
                labels=[1.0, 0.0],
                masks=[1, 0],
            ),
        ]
    )

    kwargs = {
        "enabled": True,
        "max_frames_per_sequence": 2,
        "min_distance_frames": 5,
        "sequence_gap_frames": 100,
        "split_on_label_change": False,
        "rare_label_aware": False,
    }

    old_result = (
        sample_annotations_temporally(
            old_annotations,
            **kwargs,
        )
    )

    manifest_result = (
        sample_manifest_samples_temporally(
            manifest,
            **kwargs,
        )
    )

    assert _frame_ids_from_annotations(
        old_result
    ) == _frame_ids_from_manifest_samples(
        manifest_result
    )


def test_manifest_temporal_matches_existing_sampler_on_label_change() -> None:
    """
    Verify split_on_label_change parity.

        frame 100 -> polyp
        frame 101 -> polyp
        frame 102 -> blood
        frame 103 -> blood

    Both representations must form equivalent temporal sequences.
    """

    old_annotations = [
        _annotation(
            frame_id=100,
            video_id=1,
            frame_number=10,
            label_id=10,
            value=True,
        ),
        _annotation(
            frame_id=101,
            video_id=1,
            frame_number=20,
            label_id=10,
            value=True,
        ),
        _annotation(
            frame_id=102,
            video_id=1,
            frame_number=30,
            label_id=20,
            value=True,
        ),
        _annotation(
            frame_id=103,
            video_id=1,
            frame_number=40,
            label_id=20,
            value=True,
        ),
    ]

    manifest = _manifest(
        [
            _sample(
                sample_index=0,
                frame_id=100,
                video_id=1,
                frame_number=10,
                labels=[1.0, 0.0],
                masks=[1, 0],
            ),
            _sample(
                sample_index=1,
                frame_id=101,
                video_id=1,
                frame_number=20,
                labels=[1.0, 0.0],
                masks=[1, 0],
            ),
            _sample(
                sample_index=2,
                frame_id=102,
                video_id=1,
                frame_number=30,
                labels=[0.0, 1.0],
                masks=[0, 1],
            ),
            _sample(
                sample_index=3,
                frame_id=103,
                video_id=1,
                frame_number=40,
                labels=[0.0, 1.0],
                masks=[0, 1],
            ),
        ]
    )

    kwargs = {
        "enabled": True,
        "max_frames_per_sequence": 1,
        "min_distance_frames": 1,
        "sequence_gap_frames": 100,
        "split_on_label_change": True,
        "rare_label_aware": False,
    }

    old_result = (
        sample_annotations_temporally(
            old_annotations,
            **kwargs,
        )
    )

    manifest_result = (
        sample_manifest_samples_temporally(
            manifest,
            **kwargs,
        )
    )

    assert _frame_ids_from_annotations(
        old_result
    ) == _frame_ids_from_manifest_samples(
        manifest_result
    )


def test_manifest_temporal_matches_existing_sampler_for_multilabel_frames() -> None:
    """
    Verify that a manifest frame with multiple positive labels becomes the
    same positive-label set as multiple old annotation rows on one frame.
    """

    old_annotations = [
        _annotation(
            frame_id=100,
            video_id=1,
            frame_number=10,
            label_id=10,
            value=True,
        ),
        _annotation(
            frame_id=100,
            video_id=1,
            frame_number=10,
            label_id=20,
            value=True,
        ),
        _annotation(
            frame_id=101,
            video_id=1,
            frame_number=20,
            label_id=10,
            value=True,
        ),
    ]

    manifest = _manifest(
        [
            _sample(
                sample_index=0,
                frame_id=100,
                video_id=1,
                frame_number=10,
                labels=[
                    1.0,
                    1.0,
                ],
                masks=[
                    1,
                    1,
                ],
            ),
            _sample(
                sample_index=1,
                frame_id=101,
                video_id=1,
                frame_number=20,
                labels=[
                    1.0,
                    0.0,
                ],
                masks=[
                    1,
                    0,
                ],
            ),
        ]
    )

    kwargs = {
        "enabled": True,
        "max_frames_per_sequence": 1,
        "min_distance_frames": 1,
        "sequence_gap_frames": 100,
        "split_on_label_change": True,
        "rare_label_aware": True,
    }

    old_result = (
        sample_annotations_temporally(
            old_annotations,
            **kwargs,
        )
    )

    manifest_result = (
        sample_manifest_samples_temporally(
            manifest,
            **kwargs,
        )
    )

    assert _frame_ids_from_annotations(
        old_result
    ) == _frame_ids_from_manifest_samples(
        manifest_result
    )


def test_manifest_temporal_matches_existing_sampler_across_videos() -> None:
    old_annotations = [
        _annotation(
            frame_id=100,
            video_id=1,
            frame_number=10,
            label_id=10,
            value=True,
        ),
        _annotation(
            frame_id=101,
            video_id=1,
            frame_number=20,
            label_id=10,
            value=True,
        ),
        _annotation(
            frame_id=200,
            video_id=2,
            frame_number=10,
            label_id=20,
            value=True,
        ),
        _annotation(
            frame_id=201,
            video_id=2,
            frame_number=20,
            label_id=20,
            value=True,
        ),
    ]

    manifest = _manifest(
        [
            _sample(
                sample_index=0,
                frame_id=100,
                video_id=1,
                frame_number=10,
                labels=[1.0, 0.0],
                masks=[1, 0],
            ),
            _sample(
                sample_index=1,
                frame_id=101,
                video_id=1,
                frame_number=20,
                labels=[1.0, 0.0],
                masks=[1, 0],
            ),
            _sample(
                sample_index=2,
                frame_id=200,
                video_id=2,
                frame_number=10,
                labels=[0.0, 1.0],
                masks=[0, 1],
            ),
            _sample(
                sample_index=3,
                frame_id=201,
                video_id=2,
                frame_number=20,
                labels=[0.0, 1.0],
                masks=[0, 1],
            ),
        ]
    )

    kwargs = {
        "enabled": True,
        "max_frames_per_sequence": 1,
        "min_distance_frames": 1,
        "sequence_gap_frames": 100,
        "split_on_label_change": False,
        "rare_label_aware": True,
    }

    old_result = (
        sample_annotations_temporally(
            old_annotations,
            **kwargs,
        )
    )

    manifest_result = (
        sample_manifest_samples_temporally(
            manifest,
            **kwargs,
        )
    )

    assert _frame_ids_from_annotations(
        old_result
    ) == _frame_ids_from_manifest_samples(
        manifest_result
    )


# -----------------------------------------------------------------------------
# Validation behavior
# -----------------------------------------------------------------------------


def test_manifest_temporal_requires_frame_number_when_enabled() -> None:
    manifest = _manifest(
        [
            _sample(
                sample_index=0,
                frame_id=100,
                video_id=1,
                frame_number=10,
                labels=[1.0, 0.0],
                masks=[1, 0],
            )
        ]
    )

    manifest.samples[
        0
    ].frame_number = None

    with pytest.raises(
        ValueError,
        match=(
            "frame_number is required "
            "for temporal sampling"
        ),
    ):
        sample_manifest_samples_temporally(
            manifest,
            enabled=True,
            max_frames_per_sequence=1,
            min_distance_frames=1,
            sequence_gap_frames=100,
            split_on_label_change=True,
            rare_label_aware=True,
        )


def test_manifest_temporal_does_not_require_temporal_identity_when_disabled() -> None:
    manifest = _manifest(
        [
            _sample(
                sample_index=0,
                frame_id=100,
                video_id=1,
                frame_number=10,
                labels=[1.0, 0.0],
                masks=[1, 0],
            )
        ]
    )

    manifest.samples[
        0
    ].frame_number = None

    sampled = (
        sample_manifest_samples_temporally(
            manifest,
            enabled=False,
            max_frames_per_sequence=1,
            min_distance_frames=1,
            sequence_gap_frames=100,
            split_on_label_change=True,
            rare_label_aware=True,
        )
    )

    assert len(sampled) == 1