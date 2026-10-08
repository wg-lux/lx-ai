from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import (
    AwareDatetime,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from lx_dtypes.models.base.app_base_model.pydantic.AppBaseModel import (
    AppBaseModel,
)


# -----------------------------------------------------------------------------
# Purpose
# -----------------------------------------------------------------------------
#
# This module defines LX-AI-side validation models for the serialized
# EndoRegDB training-manifest contract.
#
# IMPORTANT:
# - no endoreg_db imports
# - no PostgreSQL access
# - no Django ORM access
# - no frame materialization
# - no temporal sampling
# - no bucket assignment
# - no trainer changes
#
# EndoRegDB owns the database/domain model.
# LX-AI only consumes the serialized training contract.
# -----------------------------------------------------------------------------


AIFrameFormatStrategy = Literal[
    "preserve_dimensions_black_mask",
    "crop_to_endoscope_roi",
]


def _now_utc() -> AwareDatetime:
    return datetime.now(UTC)


def _reject_remote_path(value: str) -> None:
    """
    Match the EndoRegDB manifest rule:

    training-manifest paths are local filesystem references,
    not URLs.
    """
    if "://" in value or value.startswith("//"):
        raise ValueError("training manifest paths must be local paths")


def _coerce_local_path(value: object) -> Path | None:
    """
    Match EndoRegDB's local path validation.
    """
    if value is None or value == "":
        return None

    if not isinstance(value, (str, Path)):
        raise TypeError(
            "path must be a local path string or Path"
        )

    text = str(value).strip()

    if not text:
        return None

    _reject_remote_path(text)

    return Path(text)


def _coerce_relative_path(value: object) -> str | None:
    """
    Validate an optional relative path without permitting:
    - absolute paths
    - URL-like paths
    - parent traversal
    """
    if value is None or value == "":
        return None

    text = str(value).strip()

    if not text:
        return None

    _reject_remote_path(text)

    path = Path(text)

    if path.is_absolute():
        raise ValueError(
            "relative_path must not be absolute"
        )

    if ".." in path.parts:
        raise ValueError(
            "relative_path must not contain parent traversal"
        )

    return path.as_posix()


class EndoRegManifestBaseModel(AppBaseModel):
    """
    Base model for EndoReg -> LX-AI wire-contract objects.

    We use LX-DTypes' AppBaseModel because that is the LX-AI convention.

    AppBaseModel also provides application metadata fields such as
    source_file and created_at. Those fields are NOT part of the
    EndoReg training-manifest wire contract.

    Therefore:
    - source_file remains excluded from serialization
    - created_at is overridden and excluded
    - incoming manifest payloads are forbidden from specifying either
      internal AppBaseModel field
    """

    model_config = AppBaseModel.model_config | ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    created_at: AwareDatetime = Field(
        default_factory=_now_utc,
        exclude=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_app_base_internal_fields(
        cls,
        value: object,
    ) -> object:
        if not isinstance(value, Mapping):
            return value

        forbidden_internal_fields = {
            "source_file",
            "created_at",
        }

        supplied_internal_fields = (
            forbidden_internal_fields.intersection(value.keys())
        )

        if supplied_internal_fields:
            names = ", ".join(
                sorted(supplied_internal_fields)
            )

            raise ValueError(
                "EndoReg training-manifest payload must not "
                f"contain LX-AI internal fields: {names}"
            )

        return value


class EndoRegTrainingLabel(EndoRegManifestBaseModel):
    """
    One label from EndoRegDB's AITrainingDatasetManifest.
    """

    id: int = Field(ge=1)

    name: str = Field(min_length=1)

    index: int = Field(ge=0)

    labelset_name: str | None = None

    # Keep this compatible with the EndoRegDB manifest contract.
    # Do not impose an additional LX-AI-only numeric restriction here.
    labelset_version: int | None = None


class EndoRegTrainingSample(EndoRegManifestBaseModel):
    """
    One frame-level sample from EndoRegDB.

    The current EndoRegDB contract supports two meaningful ways
    to identify training image data:

    1. path
       A directly available local image path.

    2. video_id + frame_number
       A processed-frame reference that must later be resolved/
       streamed/materialized.

    relative_path remains represented because it is still a field
    in the EndoRegDB manifest schema, but relative_path by itself
    is NOT sufficient to make a sample usable in the current
    EndoRegDB contract.
    """

    sample_index: int = Field(ge=0)

    path: Path | None = None

    relative_path: str | None = None

    labels: list[float] = Field(
        min_length=1
    )

    label_mask: list[int] = Field(
        min_length=1
    )

    group_id: str | None = None

    frame_id: int | None = Field(
        default=None,
        ge=1,
    )

    video_id: int | None = Field(
        default=None,
        ge=1,
    )

    video_uuid: str | None = None

    frame_number: int | None = Field(
        default=None,
        ge=0,
    )

    timestamp: float | None = Field(
        default=None,
        ge=0.0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(
        cls,
        value: object,
    ) -> Path | None:
        return _coerce_local_path(value)

    @field_validator(
        "relative_path",
        mode="before",
    )
    @classmethod
    def _validate_relative_path(
        cls,
        value: object,
    ) -> str | None:
        return _coerce_relative_path(value)

    @field_validator("labels")
    @classmethod
    def _validate_labels(
        cls,
        value: list[float],
    ) -> list[float]:
        normalized = [
            float(item)
            for item in value
        ]

        invalid = [
            item
            for item in normalized
            if item < 0.0 or item > 1.0
        ]

        if invalid:
            raise ValueError(
                "labels must be probabilities "
                "in the range [0, 1]"
            )

        return normalized

    @field_validator("label_mask")
    @classmethod
    def _validate_label_mask(
        cls,
        value: list[int],
    ) -> list[int]:
        normalized = [
            int(item)
            for item in value
        ]

        invalid = [
            item
            for item in normalized
            if item not in {0, 1}
        ]

        if invalid:
            raise ValueError(
                "label_mask entries must be 0 or 1"
            )

        return normalized

    @model_validator(mode="after")
    def _validate_sample(
        self,
    ) -> "EndoRegTrainingSample":
        # This matches the current EndoRegDB schema:
        #
        # either a directly available path,
        # or a processed-frame reference
        # (video_id + frame_number).
        if self.path is None and (
            self.video_id is None
            or self.frame_number is None
        ):
            raise ValueError(
                "training sample requires either "
                "path or video_id + frame_number"
            )

        if len(self.labels) != len(
            self.label_mask
        ):
            raise ValueError(
                "label_mask must match labels length"
            )

        return self


class EndoRegFrameFormatManifest(
    EndoRegManifestBaseModel
):
    """
    Frame-format metadata produced by EndoRegDB.

    This is metadata only.

    It does NOT currently change LX-AI's existing image loading,
    resizing, normalization, or model preprocessing behavior.
    """

    check_required: bool = True

    status: Literal[
        "not_checked",
        "passed",
        "failed",
    ] = "not_checked"

    checked_frame_count: int = Field(
        default=0,
        ge=0,
    )

    expected_image_format: str | None = None

    expected_width: int | None = Field(
        default=None,
        ge=1,
    )

    expected_height: int | None = Field(
        default=None,
        ge=1,
    )

    expected_mode: str | None = None

    preprocessing_strategy: AIFrameFormatStrategy = (
        "preserve_dimensions_black_mask"
    )

    recommended_model_input_strategy: (
        AIFrameFormatStrategy
    ) = "crop_to_endoscope_roi"

    crop_templates_by_video_uuid: dict[
        str,
        list[int] | None,
    ] = Field(
        default_factory=dict
    )

    notes: list[str] = Field(
        default_factory=list
    )

    errors: list[str] = Field(
        default_factory=list
    )

    @field_validator(
        "crop_templates_by_video_uuid"
    )
    @classmethod
    def _validate_crop_templates(
        cls,
        value: dict[
            str,
            list[int] | None,
        ],
    ) -> dict[
        str,
        list[int] | None,
    ]:
        normalized: dict[
            str,
            list[int] | None,
        ] = {}

        for (
            video_uuid,
            crop_template,
        ) in value.items():
            key = str(video_uuid)

            if crop_template is None:
                normalized[key] = None
                continue

            if len(crop_template) != 4:
                raise ValueError(
                    "crop templates must be "
                    "[y1, y2, x1, x2]"
                )

            y1, y2, x1, x2 = [
                int(item)
                for item in crop_template
            ]

            if (
                y1 < 0
                or x1 < 0
                or y2 <= y1
                or x2 <= x1
            ):
                raise ValueError(
                    "crop templates must define "
                    "a positive ROI"
                )

            normalized[key] = [
                y1,
                y2,
                x1,
                x2,
            ]

        return normalized


class EndoRegTrainingDatasetManifest(
    EndoRegManifestBaseModel
):
    """
    LX-AI-side representation of EndoRegDB's
    AITrainingDatasetManifest.

    This mirrors the serialized contract.

    It deliberately does NOT import EndoRegDB models or schemas.
    """

    schema_version: Literal["1.0"] = "1.0"

    dataset_id: int | None = Field(
        default=None,
        ge=1,
    )

    name: str | None = None

    description: str | None = None

    modality: Literal["frame"] = "frame"

    task_kind: Literal[
        "multilabel_classification"
    ] = "multilabel_classification"

    labels: list[
        EndoRegTrainingLabel
    ] = Field(
        min_length=1
    )

    samples: list[
        EndoRegTrainingSample
    ] = Field(
        min_length=1
    )

    frame_format: (
        EndoRegFrameFormatManifest
    ) = Field(
        default_factory=(
            EndoRegFrameFormatManifest
        )
    )

    class_frequencies: (
        list[float] | None
    ) = None

    provenance: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    @field_validator(
        "class_frequencies"
    )
    @classmethod
    def _validate_class_frequencies(
        cls,
        value: list[float] | None,
    ) -> list[float] | None:
        if value is None:
            return None

        normalized = [
            float(item)
            for item in value
        ]

        invalid = [
            item
            for item in normalized
            if item < 0.0
            or item > 1.0
        ]

        if invalid:
            raise ValueError(
                "class_frequencies must be "
                "in the range [0, 1]"
            )

        return normalized

    @model_validator(mode="after")
    def _validate_manifest(
        self,
    ) -> "EndoRegTrainingDatasetManifest":
        label_count = len(self.labels)

        label_indices = [
            label.index
            for label in self.labels
        ]

        expected_indices = list(
            range(label_count)
        )

        if label_indices != expected_indices:
            raise ValueError(
                "labels must be indexed "
                "contiguously from 0"
            )

        for sample in self.samples:
            if (
                len(sample.labels)
                != label_count
            ):
                raise ValueError(
                    "sample labels must match "
                    "manifest labels length"
                )

            if (
                len(sample.label_mask)
                != label_count
            ):
                raise ValueError(
                    "sample label_mask must match "
                    "manifest labels length"
                )

        if (
            self.class_frequencies
            is not None
            and len(
                self.class_frequencies
            )
            != label_count
        ):
            raise ValueError(
                "class_frequencies must match "
                "manifest labels length"
            )

        return self


def validate_endoreg_training_manifest(
    payload: dict[str, Any],
) -> EndoRegTrainingDatasetManifest:
    """
    Validate an EndoRegDB training-manifest payload.

    This function deliberately performs ONLY schema validation.

    It does not:
    - access PostgreSQL
    - import EndoRegDB
    - use Django ORM
    - resolve frame files
    - materialize encrypted media
    - perform temporal sampling
    - allocate buckets
    - calculate train/val/test splits
    - start training
    """
    return (
        EndoRegTrainingDatasetManifest
        .model_validate(payload)
    )


# Rebuild nested Pydantic model references explicitly.
EndoRegTrainingDatasetManifest.model_rebuild()


__all__ = [
    "AIFrameFormatStrategy",
    "EndoRegFrameFormatManifest",
    "EndoRegTrainingDatasetManifest",
    "EndoRegTrainingLabel",
    "EndoRegTrainingSample",
    "validate_endoreg_training_manifest",
]