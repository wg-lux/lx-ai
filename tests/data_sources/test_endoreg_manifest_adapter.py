from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lx_ai.ai_model_dataset.dataset import (
    EndoMultiLabelDataset,
    MultiLabelDatasetSpec,
)
from lx_ai.data_sources.endoreg_manifest import (
    EndoRegTrainingDatasetManifest,
    validate_endoreg_training_manifest,
)
from lx_ai.data_sources.endoreg_manifest_adapter import (
    ManifestDatasetDataDict,
    build_lx_ai_dataset_from_endoreg_manifest,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_image(
    path: Path,
    *,
    size: tuple[int, int] = (32, 32),
) -> Path:
    """
    Create a real RGB image.

    A real image is important here because this test also verifies that the
    resulting adapter output is consumable by EndoMultiLabelDataset, which
    ultimately opens the image with PIL.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = Image.new(
        "RGB",
        size,
        color=(120, 80, 40),
    )

    image.save(path)

    return path.resolve()


def _manifest_payload(
    image_path: Path,
) -> dict[str, Any]:
    """
    Build a representative EndoRegDB training manifest.

    This intentionally resembles the current
    endoreg-db@lx-ai-service-prototype contract:

    - dataset_id at manifest level
    - stable contiguous label indices
    - video_id + frame_number
    - frame_id
    - label vectors + label masks
    - annotation_ids_by_label provenance
    - information_source_names
    - labelset metadata in manifest provenance
    """

    return {
        "schema_version": "1.0",
        "dataset_id": 7,
        "name": "adapter-test-dataset",
        "description": "LX-AI manifest adapter test",
        "modality": "frame",
        "task_kind": "multilabel_classification",
        "labels": [
            {
                "id": 10,
                "name": "polyp",
                "index": 0,
                "labelset_name": "test-labelset",
                "labelset_version": 2,
            },
            {
                "id": 11,
                "name": "blood",
                "index": 1,
                "labelset_name": "test-labelset",
                "labelset_version": 2,
            },
            {
                "id": 12,
                "name": "unknown-label",
                "index": 2,
                "labelset_name": "test-labelset",
                "labelset_version": 2,
            },
        ],
        "samples": [
            {
                "sample_index": 0,
                "path": str(image_path),
                "relative_path": None,
                "labels": [
                    1.0,
                    0.0,
                    0.0,
                ],
                "label_mask": [
                    1,
                    1,
                    0,
                ],
                "group_id": "video-5",
                "frame_id": 100,
                "video_id": 5,
                "video_uuid": "video-uuid-5",
                "frame_number": 250,
                "timestamp": 5.0,
                "metadata": {
                    "annotation_ids_by_label": {
                        "polyp": [5001],
                        "blood": [5002],
                    },
                    "information_source_names": [
                        "manual",
                    ],
                },
            },
        ],
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
            1.0,
            0.0,
            0.0,
        ],
        "provenance": {
            "source": "endoreg_db.AIDataSet",
            "dataset_id": 7,
            "labelset_id": 55,
            "labelset_name": "test-labelset",
            "labelset_version": 2,
            "treat_unlabeled_as_negative": False,
            "include_file_paths": False,
            "check_frame_format": False,
        },
    }


def _build_manifest(
    image_path: Path,
) -> EndoRegTrainingDatasetManifest:
    return validate_endoreg_training_manifest(
        _manifest_payload(
            image_path
        )
    )


def _resolve_manifest_path(
    sample: Any,
) -> Path:
    """
    Test-only path resolver.

    Production frame resolution is deliberately NOT implemented here.
    """

    if sample.path is None:
        raise ValueError(
            "test sample has no local path"
        )

    return sample.path


def _build_adapter_dataset(
    image_path: Path,
) -> ManifestDatasetDataDict:
    manifest = _build_manifest(
        image_path
    )

    return (
        build_lx_ai_dataset_from_endoreg_manifest(
            manifest=manifest,
            resolve_image_path=(
                _resolve_manifest_path
            ),
        )
    )


# -----------------------------------------------------------------------------
# Contract tests
# -----------------------------------------------------------------------------


def test_manifest_adapter_preserves_sample_identity(
    tmp_path: Path,
) -> None:
    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    data = _build_adapter_dataset(
        image_path
    )

    assert data["frame_ids"] == [100]

    assert data["video_ids"] == [5]

    assert data[
        "dataset_ids_per_frame"
    ] == [7]

    assert data[
        "old_examination_ids"
    ] == [None]


def test_manifest_adapter_preserves_label_order(
    tmp_path: Path,
) -> None:
    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    data = _build_adapter_dataset(
        image_path
    )

    assert [
        label["id"]
        for label in data["labels"]
    ] == [
        10,
        11,
        12,
    ]

    assert [
        label["name"]
        for label in data["labels"]
    ] == [
        "polyp",
        "blood",
        "unknown-label",
    ]


def test_manifest_adapter_preserves_unknown_semantics(
    tmp_path: Path,
) -> None:
    """
    This is one of the most important compatibility tests.

    EndoReg:
        labels=[1.0, 0.0, 0.0]
        mask=[1, 1, 0]

    must become LX-AI:
        vector=[1, 0, None]
        mask=[1, 1, 0]
    """

    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    data = _build_adapter_dataset(
        image_path
    )

    assert data[
        "label_vectors"
    ] == [
        [
            1,
            0,
            None,
        ]
    ]

    assert data[
        "label_masks"
    ] == [
        [
            1,
            1,
            0,
        ]
    ]


def test_manifest_adapter_preserves_labelset_metadata(
    tmp_path: Path,
) -> None:
    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    data = _build_adapter_dataset(
        image_path
    )

    labelset = data["labelset"]

    assert labelset["id"] == 55

    assert (
        labelset["name"]
        == "test-labelset"
    )

    assert labelset["version"] == 2

    assert [
        label["id"]
        for label in labelset["labels"]
    ] == [
        10,
        11,
        12,
    ]


def test_manifest_adapter_does_not_treat_information_sources_as_annotators(
    tmp_path: Path,
) -> None:
    """
    EndoReg information sources and the legacy SQL annotator field are not
    assumed to be semantically identical.
    """

    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    data = _build_adapter_dataset(
        image_path
    )

    assert data[
        "annotators_per_frame"
    ] == [
        []
    ]


def test_manifest_adapter_counts_explicit_annotation_rows(
    tmp_path: Path,
) -> None:
    """
    annotation_positive_count / annotation_negative_count should preserve the
    old SQL diagnostic meaning as closely as possible.

    The fixture contains:

        polyp -> one positive annotation
        blood -> one negative annotation

    The masked unknown label is not counted.
    """

    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    data = _build_adapter_dataset(
        image_path
    )

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


def test_manifest_adapter_resolves_image_path(
    tmp_path: Path,
) -> None:
    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    data = _build_adapter_dataset(
        image_path
    )

    assert data[
        "image_paths"
    ] == [
        str(
            image_path.resolve()
        )
    ]


# -----------------------------------------------------------------------------
# Existing LX-AI dataset compatibility
# -----------------------------------------------------------------------------


def test_manifest_adapter_output_is_valid_multilabel_dataset_spec(
    tmp_path: Path,
) -> None:
    """
    Verify that the important tensor-facing fields produced by the adapter are
    accepted by the existing LX-AI MultiLabelDatasetSpec.
    """

    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    data = _build_adapter_dataset(
        image_path
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

    assert len(spec.image_paths) == 1

    assert spec.label_vectors == [
        [
            1,
            0,
            None,
        ]
    ]

    assert spec.label_masks == [
        [
            1,
            1,
            0,
        ]
    ]


def test_manifest_adapter_output_is_consumable_by_torch_dataset(
    tmp_path: Path,
) -> None:
    """
    Verify that the existing EndoMultiLabelDataset can actually load an image
    produced through the manifest adapter.

    This test does not modify training behavior.
    """

    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    data = _build_adapter_dataset(
        image_path
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

    dataset = EndoMultiLabelDataset(
        spec
    )

    assert len(dataset) == 1

    image_tensor, labels, mask = (
        dataset[0]
    )

    assert image_tensor.shape == (
        3,
        64,
        64,
    )

    assert labels.tolist() == [
        1.0,
        0.0,
        0.0,
    ]

    assert mask.tolist() == [
        1.0,
        1.0,
        0.0,
    ]


# -----------------------------------------------------------------------------
# Failure-mode tests
# -----------------------------------------------------------------------------


def test_manifest_adapter_rejects_missing_frame_id(
    tmp_path: Path,
) -> None:
    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    payload = _manifest_payload(
        image_path
    )

    payload["samples"][0][
        "frame_id"
    ] = None

    manifest = (
        validate_endoreg_training_manifest(
            payload
        )
    )

    with pytest.raises(
        ValueError,
        match="frame_id is required",
    ):
        build_lx_ai_dataset_from_endoreg_manifest(
            manifest=manifest,
            resolve_image_path=(
                _resolve_manifest_path
            ),
        )


def test_manifest_adapter_rejects_missing_video_id_for_current_pipeline(
    tmp_path: Path,
) -> None:
    """
    A directly supplied path makes the manifest itself valid without video_id.

    The current LX-AI downstream pipeline, however, requires video_id for its
    active video-based bucket assignment. The adapter must fail explicitly
    rather than silently inventing one.
    """

    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    payload = _manifest_payload(
        image_path
    )

    payload["samples"][0][
        "video_id"
    ] = None

    manifest = (
        validate_endoreg_training_manifest(
            payload
        )
    )

    with pytest.raises(
        ValueError,
        match="video_id is required",
    ):
        build_lx_ai_dataset_from_endoreg_manifest(
            manifest=manifest,
            resolve_image_path=(
                _resolve_manifest_path
            ),
        )


def test_manifest_adapter_rejects_missing_resolved_image(
    tmp_path: Path,
) -> None:
    existing_image = _make_image(
        tmp_path / "frame.jpg"
    )

    manifest = _build_manifest(
        existing_image
    )

    missing = (
        tmp_path
        / "does-not-exist.jpg"
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Resolved training image "
            "does not exist"
        ),
    ):
        build_lx_ai_dataset_from_endoreg_manifest(
            manifest=manifest,
            resolve_image_path=(
                lambda sample: missing
            ),
        )


def test_manifest_adapter_does_not_filter_zero_positive_labels(
    tmp_path: Path,
) -> None:
    """
    Zero-positive filtering intentionally does NOT belong in this adapter.

    The existing PostgreSQL pipeline performs temporal sampling first and only
    later performs zero-positive-label filtering in the existing dataset
    builder.

    Keeping the unused column here protects that ordering until the manifest
    source is connected at the correct stage.
    """

    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    data = _build_adapter_dataset(
        image_path
    )

    assert [
        label["name"]
        for label in data["labels"]
    ] == [
        "polyp",
        "blood",
        "unknown-label",
    ]

    assert data[
        "label_vectors"
    ][0][2] is None

    assert data[
        "label_masks"
    ][0][2] == 0