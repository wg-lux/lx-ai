"""Mapping functions for normalizing EndoRegDB data."""

from collections.abc import Mapping
from typing import Any

from lx_ai.integrations.endoreg_db.dto import (
    AnnotationDTO,
    FrameDTO,
    LabelDTO,
    LabelSetDTO,
)
from lx_ai.integrations.endoreg_db.exceptions import (
    EndoRegDBDataError,
)


def _required_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise EndoRegDBDataError(
            f"Required integer field {key!r} is missing or invalid: {value!r}"
        )

    return value


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)

    if not isinstance(value, str) or not value.strip():
        raise EndoRegDBDataError(
            f"Required string field {key!r} is missing or invalid: {value!r}"
        )

    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, int):
        raise EndoRegDBDataError(f"Expected integer or None, got {value!r}")

    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EndoRegDBDataError(f"Expected numeric value or None, got {value!r}")

    return float(value)


def label_to_dto(data: Mapping[str, Any]) -> LabelDTO:
    return {
        "id": _required_int(data, "id"),
        "name": _required_str(data, "name"),
    }


def frame_to_dto(data: Mapping[str, Any]) -> FrameDTO:
    frame_dir = data.get("file_path") or data.get("frame_dir")

    result: FrameDTO = {
        "id": _required_int(data, "id"),
        "relative_path": data.get("relative_path"),
        "frame_number": _optional_int(data.get("frame_number")),
        "timestamp": data.get("timestamp"),
        "file_path": frame_dir,
        "frame_dir": frame_dir,
        "old_examination_id": _optional_int(data.get("old_examination_id")),
        "video_id": _required_int(data, "video_id"),
        "video_uuid": data.get("video_uuid"),
        "processed_file": data.get("processed_file"),
        "video_fps": _optional_float(data.get("video_fps")),
    }

    resolved_path = data.get("resolved_frame_path")

    if resolved_path is not None:
        if not isinstance(resolved_path, str):
            raise EndoRegDBDataError("resolved_frame_path must be a string.")

        result["resolved_frame_path"] = resolved_path

    return result


def annotation_to_dto(
    data: Mapping[str, Any],
) -> AnnotationDTO:
    frame = data.get("frame")
    label = data.get("label")

    if not isinstance(frame, Mapping):
        raise EndoRegDBDataError("Annotation does not contain valid frame data.")

    if not isinstance(label, Mapping):
        raise EndoRegDBDataError("Annotation does not contain valid label data.")

    result: AnnotationDTO = {
        "dataset_id": _required_int(data, "dataset_id"),
        "annotation_id": _required_int(data, "annotation_id"),
        "frame": frame_to_dto(frame),
        "label": label_to_dto(label),
        "value": data.get("value"),
        "annotator": data.get("annotator"),
    }

    annotation = data.get("annotation")

    if isinstance(annotation, Mapping):
        annotation_id = annotation.get("id")

        if isinstance(annotation_id, int):
            result["annotation"] = {"id": annotation_id}

    return result


def annotations_to_dto(
    rows: list[dict[str, Any]],
) -> list[AnnotationDTO]:
    return [annotation_to_dto(row) for row in rows]


def labelset_to_dto(
    data: Mapping[str, Any],
) -> LabelSetDTO:
    raw_labels = data.get("labels")

    if not isinstance(raw_labels, list) or not raw_labels:
        raise EndoRegDBDataError(
            "Labelset contains no labels or has invalid label data."
        )

    labels = []

    for label in raw_labels:
        if not isinstance(label, Mapping):
            raise EndoRegDBDataError("Labelset contains an invalid label entry.")

        labels.append(label_to_dto(label))

    return {
        "id": _required_int(data, "id"),
        "version": _required_int(data, "version"),
        "labels": labels,
    }
