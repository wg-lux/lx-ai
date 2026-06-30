from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrameGroup:
    video_id: int
    frame_id: int
    frame_number: int
    annotations: list[dict[str, Any]]
    positive_labels: frozenset[int]


def _ann_frame(ann: dict[str, Any]) -> dict[str, Any]:
    frame = ann.get("frame") or {}
    if not isinstance(frame, dict):
        return {}
    return frame


def _label_id(ann: dict[str, Any]) -> int | None:
    for key in ("label_id", "label"):
        value = ann.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, dict) and isinstance(value.get("id"), int):
            return int(value["id"])
    return None


def _is_positive(ann: dict[str, Any]) -> bool:
    value = ann.get("value", ann.get("label_value"))
    return (
        value is True or value == 1 or str(value).lower() in {"true", "positive", "1"}
    )


def _make_frame_groups(annotations: list[dict[str, Any]]) -> list[FrameGroup]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for ann in annotations:
        frame = _ann_frame(ann)
        video_id = frame.get("video_id") or ann.get("video_id")
        frame_id = frame.get("id") or frame.get("frame_id") or ann.get("frame_id")

        if video_id is None or frame_id is None:
            continue

        grouped[(int(video_id), int(frame_id))].append(ann)

    out: list[FrameGroup] = []

    for (video_id, frame_id), anns in grouped.items():
        frame = _ann_frame(anns[0])
        frame_number = frame.get("frame_number") or anns[0].get("frame_number")

        if frame_number is None:
            continue

        positives = frozenset(
            label_id
            for ann in anns
            if _is_positive(ann)
            for label_id in [_label_id(ann)]
            if label_id is not None
        )

        out.append(
            FrameGroup(
                video_id=video_id,
                frame_id=frame_id,
                frame_number=int(frame_number),
                annotations=anns,
                positive_labels=positives,
            )
        )

    return sorted(out, key=lambda g: (g.video_id, g.frame_number, g.frame_id))


def _split_sequences(
    frames: list[FrameGroup],
    *,
    sequence_gap_frames: int,
) -> list[list[FrameGroup]]:
    if not frames:
        return []

    sequences: list[list[FrameGroup]] = [[frames[0]]]

    for frame in frames[1:]:
        previous = sequences[-1][-1]
        if frame.frame_number - previous.frame_number > sequence_gap_frames:
            sequences.append([frame])
        else:
            sequences[-1].append(frame)

    return sequences


def _select_from_sequence(
    sequence: list[FrameGroup],
    *,
    max_frames_per_sequence: int,
    min_distance_frames: int,
    label_frequency: Counter[int],
) -> list[FrameGroup]:
    if len(sequence) <= max_frames_per_sequence:
        return sequence

    def score(frame: FrameGroup) -> tuple[float, int, int]:
        rare_score = sum(
            1.0 / max(label_frequency[label], 1) for label in frame.positive_labels
        )
        multi_label_score = len(frame.positive_labels)
        return (rare_score, multi_label_score, -frame.frame_number)

    candidates = sorted(sequence, key=score, reverse=True)
    selected: list[FrameGroup] = []

    for frame in candidates:
        if len(selected) >= max_frames_per_sequence:
            break

        too_close = any(
            abs(frame.frame_number - chosen.frame_number) < min_distance_frames
            for chosen in selected
        )
        if too_close:
            continue

        selected.append(frame)

    if not selected:
        selected = [sequence[len(sequence) // 2]]

    return sorted(selected, key=lambda g: (g.frame_number, g.frame_id))


def sample_annotations_temporally(
    annotations: list[dict[str, Any]],
    *,
    enabled: bool = True,
    max_frames_per_sequence: int = 5,
    min_distance_frames: int = 50,
    sequence_gap_frames: int = 250,
) -> list[dict[str, Any]]:
    """
    Frame-level temporal sampler.

    It keeps the original annotation dict format, but removes whole frames.
    For a selected frame, all annotations for that frame are preserved.

    This protects multi-label semantics:
    - no label vector is built here
    - no unknown/negative logic is changed here
    - downstream multi-label construction stays unchanged
    """
    if not enabled:
        return annotations

    if max_frames_per_sequence <= 0:
        return annotations

    groups = _make_frame_groups(annotations)
    if not groups:
        return annotations

    label_frequency: Counter[int] = Counter()
    for group in groups:
        label_frequency.update(group.positive_labels)

    by_video: dict[int, list[FrameGroup]] = defaultdict(list)
    for group in groups:
        by_video[group.video_id].append(group)

    selected_frame_ids: set[int] = set()

    for video_id, video_frames in by_video.items():
        sequences = _split_sequences(
            video_frames,
            sequence_gap_frames=sequence_gap_frames,
        )

        for sequence in sequences:
            selected = _select_from_sequence(
                sequence,
                max_frames_per_sequence=max_frames_per_sequence,
                min_distance_frames=min_distance_frames,
                label_frequency=label_frequency,
            )
            selected_frame_ids.update(frame.frame_id for frame in selected)

    sampled = [
        ann
        for ann in annotations
        if int(
            (
                _ann_frame(ann).get("id")
                or _ann_frame(ann).get("frame_id")
                or ann.get("frame_id")
            )
        )
        in selected_frame_ids
    ]

    print(
        "[LXAI SAMPLING] temporal frame sampler active: "
        f"annotations {len(annotations)} -> {len(sampled)}, "
        f"frames {len(groups)} -> {len(selected_frame_ids)}, "
        f"max_frames_per_sequence={max_frames_per_sequence}, "
        f"min_distance_frames={min_distance_frames}, "
        f"sequence_gap_frames={sequence_gap_frames}",
        flush=True,
    )

    return sampled
