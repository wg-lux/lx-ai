from typing import Any, NotRequired, TypedDict


class FrameDTO(TypedDict):
    id: int
    relative_path: str | None
    frame_number: int | None
    timestamp: Any
    file_path: str | None
    frame_dir: str | None
    old_examination_id: int | None
    video_id: int
    video_uuid: str | None
    processed_file: str | None
    video_fps: float | None
    resolved_frame_path: NotRequired[str]


class LabelDTO(TypedDict):
    id: int
    name: str


class AnnotationDTO(TypedDict):
    dataset_id: int
    annotation_id: int
    frame: FrameDTO
    label: LabelDTO
    value: Any
    annotator: Any
    annotation: NotRequired[dict[str, int]]


class LabelSetDTO(TypedDict):
    id: int
    version: int
    labels: list[LabelDTO]
