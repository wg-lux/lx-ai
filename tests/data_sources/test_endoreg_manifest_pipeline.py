from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lx_ai.ai_model_dataset.dataset import (
    MultiLabelDatasetSpec,
)
from lx_ai.data_sources.endoreg_manifest import (
    EndoRegTrainingDatasetManifest,
    validate_endoreg_training_manifest,
)
from lx_ai.data_sources.endoreg_manifest_pipeline import (
    build_lx_ai_dataset_from_endoreg_manifest_pipeline,
)
from lx_ai.utils.data_loader_for_model_training import (
    build_image_multilabel_dataset,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_image(
    path: Path,
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.new(
        "RGB",
        (32, 32),
        color=(100, 120, 140),
    ).save(path)

    return path.resolve()


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
    image_path: Path,
    frame_id: int,
    video_id: int,
    frame_number: int,
    labels: list[float],
    masks: list[int],
    annotation_ids_by_label: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "path": str(image_path),
        "relative_path": None,
        "labels": labels,
        "label_mask": masks,
        "group_id": f"video-{video_id}",
        "frame_id": frame_id,
        "video_id": video_id,
        "video_uuid": f"uuid-{video_id}",
        "frame_number": frame_number,
        "timestamp": float(frame_number),
        "metadata": {
            "annotation_ids_by_label": (
                annotation_ids_by_label
                or {}
            ),
        },
    }


def _manifest(
    *,
    samples: list[dict[str, Any]],
    labels: list[dict[str, Any]] | None = None,
) -> EndoRegTrainingDatasetManifest:
    manifest_labels = (
        labels
        if labels is not None
        else [
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
        ]
    )

    payload = {
        "schema_version": "1.0",
        "dataset_id": 7,
        "name": "pipeline-test-dataset",
        "description": None,
        "modality": "frame",
        "task_kind": "multilabel_classification",
        "labels": manifest_labels,
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
            0.0
            for _ in manifest_labels
        ],
        "provenance": {
            "source": "endoreg_db.AIDataSet",
            "dataset_id": 7,
            "labelset_id": 55,
            "labelset_name": "test-labelset",
            "labelset_version": 1,
            "treat_unlabeled_as_negative": False,
        },
    }

    return validate_endoreg_training_manifest(
        payload
    )


def _resolve_sample_path(
    sample: Any,
) -> Path:
    if sample.path is None:
        raise ValueError(
            "test sample has no path"
        )

    return sample.path


def _build_pipeline(
    manifest: EndoRegTrainingDatasetManifest,
    *,
    temporal_enabled: bool = False,
    max_frames: int = 100,
    min_distance: int = 1,
    sequence_gap: int = 1000,
    split_on_label_change: bool = False,
    rare_label_aware: bool = False,
):
    return (
        build_lx_ai_dataset_from_endoreg_manifest_pipeline(
            manifest=manifest,
            resolve_image_path=(
                _resolve_sample_path
            ),
            temporal_sampling_enabled=(
                temporal_enabled
            ),
            temporal_max_frames_per_sequence=(
                max_frames
            ),
            temporal_min_distance_frames=(
                min_distance
            ),
            temporal_sequence_gap_frames=(
                sequence_gap
            ),
            temporal_split_on_label_change=(
                split_on_label_change
            ),
            temporal_rare_label_aware=(
                rare_label_aware
            ),
        )
    )


# -----------------------------------------------------------------------------
# Basic pipeline behavior
# -----------------------------------------------------------------------------


def test_pipeline_builds_lx_ai_base_dataset(
    tmp_path: Path,
) -> None:
    frame_1 = _make_image(
        tmp_path / "frame_1.jpg"
    )

    frame_2 = _make_image(
        tmp_path / "frame_2.jpg"
    )

    manifest = _manifest(
        samples=[
            _sample(
                sample_index=0,
                image_path=frame_1,
                frame_id=100,
                video_id=5,
                frame_number=10,
                labels=[
                    1.0,
                    0.0,
                ],
                masks=[
                    1,
                    0,
                ],
                annotation_ids_by_label={
                    "polyp": [1001],
                },
            ),
            _sample(
                sample_index=1,
                image_path=frame_2,
                frame_id=101,
                video_id=5,
                frame_number=20,
                labels=[
                    0.0,
                    1.0,
                ],
                masks=[
                    0,
                    1,
                ],
                annotation_ids_by_label={
                    "blood": [1002],
                },
            ),
        ]
    )

    data = _build_pipeline(
        manifest
    )

    assert data["frame_ids"] == [
        100,
        101,
    ]

    assert data["video_ids"] == [
        5,
        5,
    ]

    assert data[
        "dataset_ids_per_frame"
    ] == [
        7,
        7,
    ]

    assert [
        label["name"]
        for label in data["labels"]
    ] == [
        "polyp",
        "blood",
    ]

    assert data["label_vectors"] == [
        [
            1,
            None,
        ],
        [
            None,
            1,
        ],
    ]

    assert data["label_masks"] == [
        [
            1,
            0,
        ],
        [
            0,
            1,
        ],
    ]


# -----------------------------------------------------------------------------
# Zero-positive filtering
# -----------------------------------------------------------------------------


def test_pipeline_drops_zero_positive_label_after_dataset_construction(
    tmp_path: Path,
) -> None:
    frame = _make_image(
        tmp_path / "frame.jpg"
    )

    manifest = _manifest(
        samples=[
            _sample(
                sample_index=0,
                image_path=frame,
                frame_id=100,
                video_id=5,
                frame_number=10,
                labels=[
                    1.0,
                    0.0,
                ],
                masks=[
                    1,
                    1,
                ],
                annotation_ids_by_label={
                    "polyp": [1001],
                    "blood": [1002],
                },
            ),
        ]
    )

    data = _build_pipeline(
        manifest
    )

    assert [
        label["name"]
        for label in data["labels"]
    ] == [
        "polyp",
    ]

    assert data["label_vectors"] == [
        [
            1,
        ]
    ]

    assert data["label_masks"] == [
        [
            1,
        ]
    ]

    assert data["labelset"][
        "labels"
    ] == [
        {
            "id": 10,
            "name": "polyp",
        }
    ]

    assert data["labelset"]["id"] == 55

    assert (
        data["labelset"]["name"]
        == "test-labelset"
    )

    assert data["labelset"]["version"] == 1


def test_pipeline_raises_when_no_positive_labels_remain(
    tmp_path: Path,
) -> None:
    frame = _make_image(
        tmp_path / "frame.jpg"
    )

    manifest = _manifest(
        samples=[
            _sample(
                sample_index=0,
                image_path=frame,
                frame_id=100,
                video_id=5,
                frame_number=10,
                labels=[
                    0.0,
                    0.0,
                ],
                masks=[
                    1,
                    1,
                ],
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "No labels with positive samples "
            "remain after filtering"
        ),
    ):
        _build_pipeline(
            manifest
        )


# -----------------------------------------------------------------------------
# Temporal sampling must happen BEFORE zero-positive filtering
# -----------------------------------------------------------------------------


def test_pipeline_filters_labels_after_temporal_sampling(
    tmp_path: Path,
) -> None:
    """
    This is the most important ordering test.

    Before temporal sampling:

        frame 100 -> polyp positive
        frame 101 -> blood positive

    Both labels therefore have positive samples.

    Temporal configuration selects only one frame from the sequence.

    With rare_label_aware=False both frames have equal score except
    frame_number; the existing sampler prefers the earlier frame.

    Therefore:

        selected frame -> 100
        remaining positive label -> polyp

    Blood must then be removed by zero-positive filtering.

    If filtering happened BEFORE temporal sampling, blood would incorrectly
    remain in the final label space.
    """

    frame_1 = _make_image(
        tmp_path / "frame_1.jpg"
    )

    frame_2 = _make_image(
        tmp_path / "frame_2.jpg"
    )

    manifest = _manifest(
        samples=[
            _sample(
                sample_index=0,
                image_path=frame_1,
                frame_id=100,
                video_id=5,
                frame_number=10,
                labels=[
                    1.0,
                    0.0,
                ],
                masks=[
                    1,
                    0,
                ],
                annotation_ids_by_label={
                    "polyp": [1001],
                },
            ),
            _sample(
                sample_index=1,
                image_path=frame_2,
                frame_id=101,
                video_id=5,
                frame_number=20,
                labels=[
                    0.0,
                    1.0,
                ],
                masks=[
                    0,
                    1,
                ],
                annotation_ids_by_label={
                    "blood": [1002],
                },
            ),
        ]
    )

    data = _build_pipeline(
        manifest,
        temporal_enabled=True,
        max_frames=1,
        min_distance=1,
        sequence_gap=100,
        split_on_label_change=False,
        rare_label_aware=False,
    )

    assert data["frame_ids"] == [
        100,
    ]

    assert [
        label["name"]
        for label in data["labels"]
    ] == [
        "polyp",
    ]

    assert data["label_vectors"] == [
        [
            1,
        ]
    ]

    assert data["label_masks"] == [
        [
            1,
        ]
    ]


def test_pipeline_without_temporal_sampling_keeps_both_positive_labels(
    tmp_path: Path,
) -> None:
    frame_1 = _make_image(
        tmp_path / "frame_1.jpg"
    )

    frame_2 = _make_image(
        tmp_path / "frame_2.jpg"
    )

    manifest = _manifest(
        samples=[
            _sample(
                sample_index=0,
                image_path=frame_1,
                frame_id=100,
                video_id=5,
                frame_number=10,
                labels=[
                    1.0,
                    0.0,
                ],
                masks=[
                    1,
                    0,
                ],
            ),
            _sample(
                sample_index=1,
                image_path=frame_2,
                frame_id=101,
                video_id=5,
                frame_number=20,
                labels=[
                    0.0,
                    1.0,
                ],
                masks=[
                    0,
                    1,
                ],
            ),
        ]
    )

    data = _build_pipeline(
        manifest,
        temporal_enabled=False,
    )

    assert [
        label["name"]
        for label in data["labels"]
    ] == [
        "polyp",
        "blood",
    ]


# -----------------------------------------------------------------------------
# Annotation diagnostic semantics
# -----------------------------------------------------------------------------


def test_pipeline_preserves_annotation_counts_before_label_filter(
    tmp_path: Path,
) -> None:
    """
    The current PostgreSQL path counts raw annotation values before its
    build_image_multilabel_dataset() call filters zero-positive labels.

    Therefore the explicit negative blood annotation remains part of the
    diagnostic negative count even though blood is removed from the final
    label space.
    """

    frame = _make_image(
        tmp_path / "frame.jpg"
    )

    manifest = _manifest(
        samples=[
            _sample(
                sample_index=0,
                image_path=frame,
                frame_id=100,
                video_id=5,
                frame_number=10,
                labels=[
                    1.0,
                    0.0,
                ],
                masks=[
                    1,
                    1,
                ],
                annotation_ids_by_label={
                    "polyp": [1001],
                    "blood": [1002],
                },
            ),
        ]
    )

    data = _build_pipeline(
        manifest
    )

    assert [
        label["name"]
        for label in data["labels"]
    ] == [
        "polyp",
    ]

    assert (
        data[
            "annotation_positive_count"
        ]
        == 1
    )

    assert (
        data[
            "annotation_negative_count"
        ]
        == 1
    )


# -----------------------------------------------------------------------------
# Existing LX-AI dataset contract
# -----------------------------------------------------------------------------


def test_pipeline_output_is_valid_multilabel_dataset_spec(
    tmp_path: Path,
) -> None:
    frame_1 = _make_image(
        tmp_path / "frame_1.jpg"
    )

    frame_2 = _make_image(
        tmp_path / "frame_2.jpg"
    )

    manifest = _manifest(
        samples=[
            _sample(
                sample_index=0,
                image_path=frame_1,
                frame_id=100,
                video_id=5,
                frame_number=10,
                labels=[
                    1.0,
                    0.0,
                ],
                masks=[
                    1,
                    0,
                ],
            ),
            _sample(
                sample_index=1,
                image_path=frame_2,
                frame_id=101,
                video_id=5,
                frame_number=20,
                labels=[
                    0.0,
                    1.0,
                ],
                masks=[
                    0,
                    1,
                ],
            ),
        ]
    )

    data = _build_pipeline(
        manifest
    )

    spec = MultiLabelDatasetSpec.model_validate(
        {
            "image_paths": (
                data["image_paths"]
            ),
            "label_vectors": (
                data["label_vectors"]
            ),
            "label_masks": (
                data["label_masks"]
            ),
            "image_size": 64,
        }
    )

    assert len(spec.image_paths) == 2

    assert spec.label_vectors == [
        [
            1,
            None,
        ],
        [
            None,
            1,
        ],
    ]

    assert spec.label_masks == [
        [
            1,
            0,
        ],
        [
            0,
            1,
        ],
    ]


# -----------------------------------------------------------------------------
# Parity with existing build_image_multilabel_dataset()
# -----------------------------------------------------------------------------


def test_manifest_pipeline_matches_existing_dataset_builder_base_semantics(
    tmp_path: Path,
) -> None:
    """
    Compare the new manifest pipeline against the existing LX-AI annotation
    builder for the same semantic dataset.

    This deliberately compares training-critical base fields rather than
    implementation-specific source metadata.
    """

    image_dir = tmp_path / "images"

    frame_1 = _make_image(
        image_dir / "frame_1.jpg"
    )

    frame_2 = _make_image(
        image_dir / "frame_2.jpg"
    )

    manifest = _manifest(
        samples=[
            _sample(
                sample_index=0,
                image_path=frame_1,
                frame_id=100,
                video_id=5,
                frame_number=10,
                labels=[
                    1.0,
                    0.0,
                ],
                masks=[
                    1,
                    0,
                ],
            ),
            _sample(
                sample_index=1,
                image_path=frame_2,
                frame_id=101,
                video_id=5,
                frame_number=20,
                labels=[
                    0.0,
                    1.0,
                ],
                masks=[
                    0,
                    1,
                ],
            ),
        ]
    )

    manifest_data = _build_pipeline(
        manifest,
        temporal_enabled=False,
    )

    existing_labelset = {
        "id": 55,
        "name": "test-labelset",
        "version": 1,
        "labels": [
            {
                "id": 10,
                "name": "polyp",
            },
            {
                "id": 20,
                "name": "blood",
            },
        ],
    }

    existing_annotations = [
        {
            "dataset_id": 7,
            "frame": {
                "id": 100,
                "file_path": str(
                    image_dir
                ),
                "relative_path": (
                    "frame_1.jpg"
                ),
                "frame_number": 10,
                "old_examination_id": None,
                "video_id": 5,
            },
            "label": {
                "id": 10,
                "name": "polyp",
            },
            "value": True,
        },
        {
            "dataset_id": 7,
            "frame": {
                "id": 101,
                "file_path": str(
                    image_dir
                ),
                "relative_path": (
                    "frame_2.jpg"
                ),
                "frame_number": 20,
                "old_examination_id": None,
                "video_id": 5,
            },
            "label": {
                "id": 20,
                "name": "blood",
            },
            "value": True,
        },
    ]

    existing_data = (
        build_image_multilabel_dataset(
            dataset_uuid=(
                "pipeline-parity-test"
            ),
            annotations=(
                existing_annotations
            ),
            labelset=(
                existing_labelset
            ),
            treat_unlabeled_as_negative=False,
        )
    )

    assert manifest_data[
        "frame_ids"
    ] == existing_data[
        "frame_ids"
    ]

    assert manifest_data[
        "video_ids"
    ] == existing_data[
        "video_ids"
    ]

    assert manifest_data[
        "dataset_ids_per_frame"
    ] == existing_data[
        "dataset_ids_per_frame"
    ]

    assert manifest_data[
        "label_vectors"
    ] == existing_data[
        "label_vectors"
    ]

    assert manifest_data[
        "label_masks"
    ] == existing_data[
        "label_masks"
    ]

    assert [
        label["id"]
        for label
        in manifest_data["labels"]
    ] == [
        label["id"]
        for label
        in existing_data["labels"]
    ]

    assert [
        label["name"]
        for label
        in manifest_data["labels"]
    ] == [
        label["name"]
        for label
        in existing_data["labels"]
    ]


# -----------------------------------------------------------------------------
# Resolver boundary
# -----------------------------------------------------------------------------


def test_pipeline_resolves_only_samples_remaining_after_temporal_sampling(
    tmp_path: Path,
) -> None:
    """
    Frame resolution/materialization should happen after temporal sampling.

    This is important for production because we should not fetch/materialize
    frames that the temporal sampler has already removed.
    """

    frame_1 = _make_image(
        tmp_path / "frame_1.jpg"
    )

    frame_2 = _make_image(
        tmp_path / "frame_2.jpg"
    )

    manifest = _manifest(
        samples=[
            _sample(
                sample_index=0,
                image_path=frame_1,
                frame_id=100,
                video_id=5,
                frame_number=10,
                labels=[
                    1.0,
                    0.0,
                ],
                masks=[
                    1,
                    0,
                ],
            ),
            _sample(
                sample_index=1,
                image_path=frame_2,
                frame_id=101,
                video_id=5,
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

    resolved_frame_ids: list[int] = []

    def resolver(
        sample: Any,
    ) -> Path:
        assert sample.frame_id is not None

        resolved_frame_ids.append(
            sample.frame_id
        )

        assert sample.path is not None

        return sample.path

    data = (
        build_lx_ai_dataset_from_endoreg_manifest_pipeline(
            manifest=manifest,
            resolve_image_path=resolver,
            temporal_sampling_enabled=True,
            temporal_max_frames_per_sequence=1,
            temporal_min_distance_frames=1,
            temporal_sequence_gap_frames=100,
            temporal_split_on_label_change=False,
            temporal_rare_label_aware=False,
        )
    )

    assert data["frame_ids"] == [
        100,
    ]

    assert resolved_frame_ids == [
        100,
    ]