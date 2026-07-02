from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any
from lx_ai.utils.logging_utils import section, subsection, kv, table_header, soft_line


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
    split_on_label_change: bool,
) -> list[list[FrameGroup]]:
    if not frames:
        return []

    sequences: list[list[FrameGroup]] = [[frames[0]]]

    for frame in frames[1:]:
        previous = sequences[-1][-1]

        time_gap_break = (
            frame.frame_number - previous.frame_number > sequence_gap_frames
        )

        label_break = (
            split_on_label_change and frame.positive_labels != previous.positive_labels
        )

        if time_gap_break or label_break:
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
    rare_label_aware: bool,
) -> list[FrameGroup]:
    if len(sequence) <= max_frames_per_sequence:
        return sequence

    def score(frame: FrameGroup) -> tuple[float, int, int]:
        rare_score = (
            sum(1.0 / max(label_frequency[label], 1) for label in frame.positive_labels)
            if rare_label_aware
            else 0.0
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
    enabled: bool,
    max_frames_per_sequence: int,
    min_distance_frames: int,
    sequence_gap_frames: int,
    split_on_label_change: bool,
    rare_label_aware: bool,
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
            split_on_label_change=split_on_label_change,
        )

        for sequence in sequences:
            selected = _select_from_sequence(
                sequence,
                max_frames_per_sequence=max_frames_per_sequence,
                min_distance_frames=min_distance_frames,
                label_frequency=label_frequency,
                rare_label_aware=rare_label_aware,
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

    total_sequences = 0
    for video_frames in by_video.values():
        total_sequences += len(
            _split_sequences(
                video_frames,
                sequence_gap_frames=sequence_gap_frames,
                split_on_label_change=split_on_label_change,
            )
        )

    section("TEMPORAL FRAME SAMPLING", icon="🎞️")

    subsection("CONFIGURATION")
    kv("Enabled", enabled)
    kv("Max frames/sequence", max_frames_per_sequence)
    kv("Min distance frames", min_distance_frames)
    kv("Sequence gap frames", sequence_gap_frames)
    kv("Split on label change", split_on_label_change)
    kv("Rare label aware", rare_label_aware)

    subsection("MEANING")
    print(
        "• enabled: activates frame-level temporal sampling before materialization/training"
    )
    print(
        "• max_frames_per_sequence: maximum selected frames from one continuous sequence"
    )
    print(
        "• min_distance_frames: minimum frame-number distance between selected frames"
    )
    print("• sequence_gap_frames: gap after which a new temporal sequence starts")
    print(
        "• split_on_label_change: starts a new sequence when positive-label set changes"
    )
    print("• rare_label_aware: prioritizes rare positive labels and multi-label frames")

    subsection("SAMPLING SUMMARY")
    kv("Input annotations", len(annotations))
    kv("Output annotations", len(sampled))
    kv("Input unique frames", len(groups))
    kv("Selected unique frames", len(selected_frame_ids))
    kv("Detected sequences", total_sequences)
    kv("Videos", len(by_video))

    subsection("REDUCTION")
    annotation_reduction = 100.0 * (1.0 - (len(sampled) / max(len(annotations), 1)))
    frame_reduction = 100.0 * (1.0 - (len(selected_frame_ids) / max(len(groups), 1)))

    kv("Annotation reduction", f"{annotation_reduction:.2f}%")
    kv("Frame reduction", f"{frame_reduction:.2f}%")

    soft_line()

    subsection("PER-VIDEO SAMPLING")
    table_header("Video", "Frames", "Seq", "Selected")

    for video_id, video_frames in sorted(by_video.items()):
        sequences = _split_sequences(
            video_frames,
            sequence_gap_frames=sequence_gap_frames,
            split_on_label_change=split_on_label_change,
        )
        selected_count = sum(
            1 for frame in video_frames if frame.frame_id in selected_frame_ids
        )
        print(
            f"{video_id:<10} "
            f"{len(video_frames):<10} "
            f"{len(sequences):<10} "
            f"{selected_count:<10}"
        )

    return sampled
