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
# Convert the validated EndoRegDB manifest into the dataset shape already used
# by LX-AI.
#
# IMPORTANT:
# - no PostgreSQL access
# - no EndoRegDB ORM imports
# - no Django ORM access
# - no bucket allocation
# - no temporal sampling
# - no train/val/test split calculation
# - no trainer changes
#
# Image resolution is injected separately because the existing LX-AI trainer
# still ultimately needs a readable local file path.
# -----------------------------------------------------------------------------


class LabelInfo(TypedDict):
    id: int
    name: str


class LabelSetInfo(TypedDict, total=False):
    name: str
    version: int
    labels: list[LabelInfo]


class ManifestDatasetDataDict(TypedDict):
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


ImagePathResolver = Callable[
    [EndoRegTrainingSample],
    Path,
]


def _convert_label_value(
    *,
    value: float,
    mask: int,
    sample_index: int,
    label_index: int,
) -> Optional[int]:
    """
    Convert EndoReg manifest label representation into the exact semantic
    representation expected by the current LX-AI training pipeline.

    Current LX-AI semantics:

        value=None, mask=0  -> unknown
        value=0,    mask=1  -> known negative
        value=1,    mask=1  -> known positive
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
        "current LX-AI multilabel training requires "
        f"binary labels 0.0 or 1.0 when mask=1, got {value!r}"
    )


def _build_label_info(
    manifest: EndoRegTrainingDatasetManifest,
) -> list[LabelInfo]:
    return [
        {
            "id": label.id,
            "name": label.name,
        }
        for label in manifest.labels
    ]


def _build_labelset_info(
    manifest: EndoRegTrainingDatasetManifest,
    labels: list[LabelInfo],
) -> LabelSetInfo:
    """
    Reconstruct the labelset-shaped metadata expected by the existing LX-AI
    trainer.

    EndoRegDB currently carries labelset name/version on each manifest label.
    All labels must agree.
    """
    first_label = manifest.labels[0]

    labelset_name = first_label.labelset_name
    labelset_version = first_label.labelset_version

    for label in manifest.labels:
        if label.labelset_name != labelset_name:
            raise ValueError(
                "EndoReg manifest contains labels from "
                "different labelset names"
            )

        if label.labelset_version != labelset_version:
            raise ValueError(
                "EndoReg manifest contains labels from "
                "different labelset versions"
            )

    result: LabelSetInfo = {
        "labels": labels,
    }

    if labelset_name is not None:
        result["name"] = labelset_name

    if labelset_version is not None:
        result["version"] = labelset_version

    return result


def _get_old_examination_id(
    sample: EndoRegTrainingSample,
) -> Optional[int]:
    """
    Legacy compatibility only.

    old_examination_id is not part of the modern required manifest contract.
    If EndoRegDB explicitly provides it in metadata, preserve it.
    Otherwise return None.
    """
    value = sample.metadata.get(
        "old_examination_id"
    )

    if value is None:
        return None

    if not isinstance(value, int):
        raise ValueError(
            f"sample={sample.sample_index}: "
            "metadata['old_examination_id'] "
            "must be int or null"
        )

    return value


def _get_annotators(
    sample: EndoRegTrainingSample,
) -> list[Any]:
    """
    Preserve annotator metadata only when explicitly present.

    We do not invent or infer annotators.
    """
    value = sample.metadata.get(
        "annotators"
    )

    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(
            f"sample={sample.sample_index}: "
            "metadata['annotators'] must be a list"
        )

    return list(value)


def build_lx_ai_dataset_from_endoreg_manifest(
    *,
    manifest: EndoRegTrainingDatasetManifest,
    resolve_image_path: ImagePathResolver,
) -> ManifestDatasetDataDict:
    """
    Convert EndoRegDB's validated manifest into the base dataset shape
    expected by LX-AI.

    This deliberately stops before:
    - temporal sampling
    - bucket allocation
    - split generation
    - training

    Those remain owned by the existing LX-AI logic.
    """

    if manifest.dataset_id is None:
        raise ValueError(
            "EndoReg training manifest must contain dataset_id "
            "for current LX-AI compatibility"
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

    positive_count = 0
    negative_count = 0

    for sample in manifest.samples:
        # -------------------------------------------------------------
        # Current LX-AI downstream code requires frame/video identity.
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # Resolve to the actual readable image required by
        # EndoMultiLabelDataset / PIL.Image.open().
        #
        # The resolver is deliberately external so this adapter does not
        # depend on EndoRegDB storage/encryption internals.
        # -------------------------------------------------------------
        resolved_path = resolve_image_path(
            sample
        )

        if not isinstance(
            resolved_path,
            Path,
        ):
            raise TypeError(
                "resolve_image_path() must return pathlib.Path"
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

        # -------------------------------------------------------------
        # Convert labels while preserving current mask semantics.
        # -------------------------------------------------------------
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

            if mask == 1:
                if converted == 1:
                    positive_count += 1
                elif converted == 0:
                    negative_count += 1

        label_vectors.append(
            vector
        )

        label_masks.append(
            mask_row
        )

        # -------------------------------------------------------------
        # Preserve aligned metadata arrays used by existing LX-AI code.
        # -------------------------------------------------------------
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
    # Defensive alignment validation.
    # -----------------------------------------------------------------
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

    invalid = {
        key: value
        for key, value
        in lengths.items()
        if value != sample_count
    }

    if invalid:
        raise ValueError(
            "Manifest adapter produced misaligned arrays. "
            f"Expected each length={sample_count}, "
            f"got {invalid}"
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
            positive_count
        ),
        "annotation_negative_count": (
            negative_count
        ),
    }


__all__ = [
    "ImagePathResolver",
    "ManifestDatasetDataDict",
    "build_lx_ai_dataset_from_endoreg_manifest",
]