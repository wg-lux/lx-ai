from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator

from lx_dtypes.models.base.app_base_model.pydantic.AppBaseModel import AppBaseModel

# -----------------------------------------------------------------------------
# Purpose
# -----------------------------------------------------------------------------
#
# These models describe the EndoRegDB training-manifest payload inside LX-AI.
#
# IMPORTANT:
# - This module does NOT import endoreg_db.
# - This module does NOT access PostgreSQL.
# - This module does NOT access Django ORM.
# - This module does NOT change the current training pipeline.
#
# It is only a validated boundary between EndoRegDB and LX-AI.
# -----------------------------------------------------------------------------


class EndoRegTrainingLabel(BaseModel):
    """
    One label from EndoRegDB's AITrainingDatasetManifest.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: int = Field(ge=1)
    name: str = Field(min_length=1)
    index: int = Field(ge=0)

    labelset_name: str | None = None
    labelset_version: int | None = Field(default=None, ge=1)


class EndoRegTrainingSample(BaseModel):
    """
    One frame-level training sample.

    We deliberately support all access forms currently relevant during migration:

    1. path
       Older/local path-based manifest behavior.

    2. relative_path
       Older EndoReg manifest compatibility.

    3. video_id + frame_number
       New processed-frame reference behavior.

    LX-AI will later resolve these into the local image path required by the
    existing PyTorch dataset. That resolution is NOT done in this model.
    """

    model_config = ConfigDict(extra="forbid")

    sample_index: int = Field(ge=0)

    path: Path | None = None
    relative_path: str | None = None

    labels: list[float] = Field(min_length=1)
    label_mask: list[int] = Field(min_length=1)

    group_id: str | None = None

    frame_id: int | None = Field(default=None, ge=1)
    video_id: int | None = Field(default=None, ge=1)
    video_uuid: str | None = None

    frame_number: int | None = Field(default=None, ge=0)
    timestamp: float | None = Field(default=None, ge=0.0)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: object) -> Path | None:
        if value is None or value == "":
            return None

        if isinstance(value, Path):
            return value

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            return Path(value)

        raise TypeError("path must be str, Path, or None")

    @field_validator("relative_path", mode="before")
    @classmethod
    def _validate_relative_path(cls, value: object) -> str | None:
        if value is None or value == "":
            return None

        text = str(value).strip()
        if not text:
            return None

        path = Path(text)

        if path.is_absolute():
            raise ValueError("relative_path must not be absolute")

        if ".." in path.parts:
            raise ValueError("relative_path must not contain parent traversal")

        return path.as_posix()

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, value: list[float]) -> list[float]:
        normalized = [float(item) for item in value]

        for item in normalized:
            if item < 0.0 or item > 1.0:
                raise ValueError(
                    "labels entries must be in the range [0, 1]"
                )

        return normalized

    @field_validator("label_mask")
    @classmethod
    def _validate_label_mask(cls, value: list[int]) -> list[int]:
        normalized = [int(item) for item in value]

        for item in normalized:
            if item not in (0, 1):
                raise ValueError(
                    "label_mask entries must be 0 or 1"
                )

        return normalized

    @model_validator(mode="after")
    def _validate_sample(self) -> "EndoRegTrainingSample":
        if len(self.labels) != len(self.label_mask):
            raise ValueError(
                "label_mask length must match labels length"
            )

        has_local_reference = (
            self.path is not None
            or self.relative_path is not None
        )

        has_processed_frame_reference = (
            self.video_id is not None
            and self.frame_number is not None
        )

        if not has_local_reference and not has_processed_frame_reference:
            raise ValueError(
                "training sample requires either "
                "path/relative_path or video_id + frame_number"
            )

        return self


AIFrameFormatStrategy = Literal[
    "preserve_dimensions_black_mask",
    "crop_to_endoscope_roi",
]


class EndoRegFrameFormatManifest(BaseModel):
    """
    Frame-format metadata supplied by EndoRegDB.

    This is metadata only for now. It does not alter existing LX-AI image
    preprocessing.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    check_required: bool = True

    status: Literal[
        "not_checked",
        "passed",
        "failed",
    ] = "not_checked"

    checked_frame_count: int = Field(default=0, ge=0)

    expected_image_format: str | None = None
    expected_width: int | None = Field(default=None, ge=1)
    expected_height: int | None = Field(default=None, ge=1)
    expected_mode: str | None = None

    preprocessing_strategy: AIFrameFormatStrategy = (
        "preserve_dimensions_black_mask"
    )

    recommended_model_input_strategy: AIFrameFormatStrategy = (
        "crop_to_endoscope_roi"
    )

    crop_templates_by_video_uuid: dict[
        str,
        list[int] | None,
    ] = Field(default_factory=dict)

    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @field_validator("crop_templates_by_video_uuid")
    @classmethod
    def _validate_crop_templates(
        cls,
        value: dict[str, list[int] | None],
    ) -> dict[str, list[int] | None]:
        normalized: dict[str, list[int] | None] = {}

        for video_uuid, crop_template in value.items():
            key = str(video_uuid)

            if crop_template is None:
                normalized[key] = None
                continue

            if len(crop_template) != 4:
                raise ValueError(
                    "crop templates must be [y1, y2, x1, x2]"
                )

            y1, y2, x1, x2 = [
                int(item) for item in crop_template
            ]

            if (
                y1 < 0
                or x1 < 0
                or y2 <= y1
                or x2 <= x1
            ):
                raise ValueError(
                    "crop templates must define a positive ROI"
                )

            normalized[key] = [y1, y2, x1, x2]

        return normalized


class EndoRegTrainingDatasetManifest(BaseModel):
    """
    LX-AI representation of EndoRegDB's training manifest.

    This deliberately mirrors the wire contract rather than importing
    EndoRegDB's Pydantic classes.

    That is important because EndoRegDB owns its ORM/models while LX-AI only
    consumes the serialized training contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    schema_version: Literal["1.0"] = "1.0"

    dataset_id: int | None = Field(default=None, ge=1)

    name: str | None = None
    description: str | None = None

    modality: Literal["frame"] = "frame"

    task_kind: Literal[
        "multilabel_classification"
    ] = "multilabel_classification"

    labels: list[EndoRegTrainingLabel] = Field(
        min_length=1
    )

    samples: list[EndoRegTrainingSample] = Field(
        min_length=1
    )

    frame_format: EndoRegFrameFormatManifest = Field(
        default_factory=EndoRegFrameFormatManifest
    )

    class_frequencies: list[float] | None = None

    provenance: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator("class_frequencies")
    @classmethod
    def _validate_class_frequencies(
        cls,
        value: list[float] | None,
    ) -> list[float] | None:
        if value is None:
            return None

        normalized = [float(item) for item in value]

        for item in normalized:
            if item < 0.0 or item > 1.0:
                raise ValueError(
                    "class_frequencies entries must be "
                    "in the range [0, 1]"
                )

        return normalized

    @model_validator(mode="after")
    def _validate_manifest(
        self,
    ) -> "EndoRegTrainingDatasetManifest":
        label_count = len(self.labels)

        # EndoReg manifest promises stable contiguous label order:
        # 0, 1, 2, ... N-1
        expected_indices = list(range(label_count))
        actual_indices = [
            label.index for label in self.labels
        ]

        if actual_indices != expected_indices:
            raise ValueError(
                "manifest labels must be indexed "
                "contiguously from 0"
            )

        if self.class_frequencies is not None:
            if len(self.class_frequencies) != label_count:
                raise ValueError(
                    "class_frequencies length must match "
                    "manifest labels length"
                )

        seen_sample_indices: set[int] = set()

        for sample in self.samples:
            if len(sample.labels) != label_count:
                raise ValueError(
                    f"sample {sample.sample_index}: labels length "
                    "must match manifest labels length"
                )

            if len(sample.label_mask) != label_count:
                raise ValueError(
                    f"sample {sample.sample_index}: label_mask "
                    "length must match manifest labels length"
                )

            if sample.sample_index in seen_sample_indices:
                raise ValueError(
                    f"duplicate sample_index: "
                    f"{sample.sample_index}"
                )

            seen_sample_indices.add(sample.sample_index)

        return self


def validate_endoreg_training_manifest(
    payload: dict[str, Any],
) -> EndoRegTrainingDatasetManifest:
    """
    Validate an EndoRegDB training-manifest JSON payload.

    No database access occurs here.

    No frame materialization occurs here.

    No temporal sampling occurs here.

    No bucket assignment occurs here.

    No training behavior is changed here.
    """
    return EndoRegTrainingDatasetManifest.model_validate(
        payload
    )


__all__ = [
    "AIFrameFormatStrategy",
    "EndoRegFrameFormatManifest",
    "EndoRegTrainingDatasetManifest",
    "EndoRegTrainingLabel",
    "EndoRegTrainingSample",
    "validate_endoreg_training_manifest",
]