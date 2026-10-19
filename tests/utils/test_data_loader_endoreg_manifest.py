from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lx_ai.ai_model_config.config import (
    TrainingConfig,
)
from lx_ai.data_sources.endoreg_manifest import (
    EndoRegTrainingDatasetManifest,
    validate_endoreg_training_manifest,
)
from lx_ai.utils.data_loader_for_model_input import (
    build_dataset_for_training,
)


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


def _manifest(
    image_path: Path,
) -> EndoRegTrainingDatasetManifest:
    return validate_endoreg_training_manifest(
        {
            "schema_version": "1.0",
            "dataset_id": 7,
            "name": "central-loader-test",
            "description": None,
            "modality": "frame",
            "task_kind": (
                "multilabel_classification"
            ),
            "labels": [
                {
                    "id": 10,
                    "name": "polyp",
                    "index": 0,
                    "labelset_name": (
                        "test-labelset"
                    ),
                    "labelset_version": 1,
                },
            ],
            "samples": [
                {
                    "sample_index": 0,
                    "path": str(
                        image_path
                    ),
                    "relative_path": None,
                    "labels": [1.0],
                    "label_mask": [1],
                    "group_id": "video-5",
                    "frame_id": 100,
                    "video_id": 5,
                    "video_uuid": (
                        "video-uuid-5"
                    ),
                    "frame_number": 10,
                    "timestamp": 1.0,
                    "metadata": {
                        "annotation_ids_by_label": {
                            "polyp": [
                                1001
                            ],
                        },
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
                1.0
            ],
            "provenance": {
                "source": (
                    "endoreg_db.AIDataSet"
                ),
                "dataset_id": 7,
                "labelset_id": 55,
                "labelset_name": (
                    "test-labelset"
                ),
                "labelset_version": 1,
            },
        }
    )


def _config(
    tmp_path: Path,
) -> TrainingConfig:
    return TrainingConfig.model_validate(
        {
            "dataset_uuid": (
                "central-loader-test"
            ),
            "data_source": (
                "endoreg_manifest"
            ),
            "dataset_ids": None,
            "labelset_id": None,
            "labelset_version_to_train": 1,
            "treat_unlabeled_as_negative": (
                False
            ),
            "base_dir": str(
                tmp_path
            ),
            "training_root": str(
                tmp_path / "training"
            ),
            "checkpoints_dir": str(
                tmp_path
                / "training"
                / "checkpoints"
            ),
            "runs_dir": str(
                tmp_path
                / "training"
                / "runs"
            ),
            "create_dirs": True,
            "backbone_name": (
                "resnet50_random"
            ),
            "backbone_checkpoint": None,
            "freeze_backbone": True,
            "num_epochs": 1,
            "batch_size": 1,
            "lr_head": 1e-3,
            "lr_backbone": 1e-4,
            "gamma_focal": 2.0,
            "alpha_focal": 0.25,
            "use_scheduler": False,
            "warmup_epochs": 0,
            "min_lr": 1e-6,
            "device": "cpu",
            "random_seed": 42,
            "temporal_sampling": {
                "enabled": False,
                "max_frames_per_sequence": 5,
                "min_distance_frames": 50,
                "sequence_gap_frames": 250,
                "split_on_label_change": True,
                "rare_label_aware": True,
            },
            "bucket_policy": {
                "num_buckets": 5,
                "validation_buckets": [
                    3
                ],
                "test_buckets": [
                    4
                ],
            },
            "save_bucket_snapshot": False,
        }
    )


def _resolve_path(
    sample: Any,
) -> Path:
    if sample.path is None:
        raise ValueError(
            "test sample has no path"
        )

    return sample.path


def test_build_dataset_for_training_accepts_endoreg_manifest(
    tmp_path: Path,
) -> None:
    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    manifest = _manifest(
        image_path
    )

    config = _config(
        tmp_path
    )

    data = build_dataset_for_training(
        config=config,
        manifest=manifest,
        image_path_resolver=(
            _resolve_path
        ),
    )

    assert data["frame_ids"] == [
        100
    ]

    assert data["video_ids"] == [
        5
    ]

    assert data[
        "dataset_ids_per_frame"
    ] == [
        7
    ]

    assert data["label_vectors"] == [
        [
            1
        ]
    ]

    assert data["label_masks"] == [
        [
            1
        ]
    ]

    assert data[
        "annotation_positive_count"
    ] == 1

    assert data[
        "annotation_negative_count"
    ] == 0

    assert len(
        data["train_indices"]
        + data["val_indices"]
        + data["test_indices"]
    ) == 1

    assert len(
        data["bucket_ids_per_sample"]
    ) == 1

    assert (
        "allocation_diagnostics"
        in data
    )


def test_build_dataset_for_training_requires_manifest(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match=(
            "manifest must be provided"
        ),
    ):
        build_dataset_for_training(
            config=config,
            manifest=None,
            image_path_resolver=(
                _resolve_path
            ),
        )


def test_build_dataset_for_training_requires_image_path_resolver(
    tmp_path: Path,
) -> None:
    image_path = _make_image(
        tmp_path / "frame.jpg"
    )

    manifest = _manifest(
        image_path
    )

    config = _config(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match=(
            "image_path_resolver "
            "must be provided"
        ),
    ):
        build_dataset_for_training(
            config=config,
            manifest=manifest,
            image_path_resolver=None,
        )