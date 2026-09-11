from __future__ import annotations

from lx_ai.utils.temporal_frame_sampler import sample_annotations_temporally

"""
Tests for the temporal frame sampler.

These tests verify that the sampler:

- preserves the existing annotation dictionary structure
- samples complete frames (never partial labels)
- preserves multi-label annotations
- works for small and large videos
- splits sequences correctly
- respects temporal spacing
- behaves correctly when disabled
- remains compatible with the existing lx-ai pipeline

These tests intentionally use synthetic annotation dictionaries so they
do not require PostgreSQL, FFmpeg, encrypted storage, or real videos.
"""


def ann(
    *,
    annotation_id: int,
    frame_id: int,
    video_id: int,
    frame_number: int,
    label_id: int,
    value: bool | None = True,
) -> dict:
    return {
        "annotation_id": annotation_id,
        "value": value,
        "label": {"id": label_id, "name": f"label_{label_id}"},
        "frame": {
            "id": frame_id,
            "video_id": video_id,
            "frame_number": frame_number,
        },
    }


def frame_ids(rows: list[dict]) -> set[int]:
    return {int(row["frame"]["id"]) for row in rows}


def test_sampler_disabled_returns_original_annotations():
    rows = [
        ann(annotation_id=1, frame_id=10, video_id=1, frame_number=100, label_id=1),
        ann(annotation_id=2, frame_id=11, video_id=1, frame_number=101, label_id=1),
    ]

    sampled = sample_annotations_temporally(
        rows,
        enabled=False,
        max_frames_per_sequence=5,
        min_distance_frames=50,
        sequence_gap_frames=250,
        split_on_label_change=True,
        rare_label_aware=True,
    )

    assert sampled == rows


def test_sampler_keeps_all_annotations_for_selected_frame():
    rows = [
        ann(annotation_id=1, frame_id=10, video_id=1, frame_number=100, label_id=1),
        ann(annotation_id=2, frame_id=10, video_id=1, frame_number=100, label_id=2),
        ann(annotation_id=3, frame_id=11, video_id=1, frame_number=101, label_id=1),
    ]

    sampled = sample_annotations_temporally(
        rows,
        enabled=True,
        max_frames_per_sequence=1,
        min_distance_frames=1,
        sequence_gap_frames=999,
        split_on_label_change=False,
        rare_label_aware=True,
    )

    selected = frame_ids(sampled)
    assert len(selected) == 1

    selected_frame_id = next(iter(selected))
    selected_rows = [row for row in sampled if row["frame"]["id"] == selected_frame_id]

    # If a frame is selected, all labels/annotations for that frame must remain together.
    if selected_frame_id == 10:
        assert {row["label"]["id"] for row in selected_rows} == {1, 2}


def test_sampler_limits_frames_per_sequence():
    rows = [
        ann(
            annotation_id=i,
            frame_id=1000 + i,
            video_id=1,
            frame_number=100 + i,
            label_id=1,
        )
        for i in range(20)
    ]

    sampled = sample_annotations_temporally(
        rows,
        enabled=True,
        max_frames_per_sequence=5,
        min_distance_frames=1,
        sequence_gap_frames=999,
        split_on_label_change=False,
        rare_label_aware=True,
    )

    assert len(frame_ids(sampled)) <= 5


def test_sampler_does_not_break_when_sequence_has_fewer_frames_than_limit():
    rows = [
        ann(annotation_id=1, frame_id=10, video_id=1, frame_number=100, label_id=1),
        ann(annotation_id=2, frame_id=11, video_id=1, frame_number=101, label_id=1),
    ]

    sampled = sample_annotations_temporally(
        rows,
        enabled=True,
        max_frames_per_sequence=5,
        min_distance_frames=50,
        sequence_gap_frames=250,
        split_on_label_change=False,
        rare_label_aware=True,
    )

    assert frame_ids(sampled) == {10, 11}


def test_sampler_splits_sequences_by_temporal_gap():
    rows = [
        ann(annotation_id=1, frame_id=10, video_id=1, frame_number=100, label_id=1),
        ann(annotation_id=2, frame_id=11, video_id=1, frame_number=101, label_id=1),
        ann(annotation_id=3, frame_id=12, video_id=1, frame_number=1000, label_id=1),
        ann(annotation_id=4, frame_id=13, video_id=1, frame_number=1001, label_id=1),
    ]

    sampled = sample_annotations_temporally(
        rows,
        enabled=True,
        max_frames_per_sequence=1,
        min_distance_frames=1,
        sequence_gap_frames=250,
        split_on_label_change=False,
        rare_label_aware=True,
    )

    # Two temporal sequences, one frame selected from each.
    assert len(frame_ids(sampled)) == 2


def test_sampler_splits_sequences_by_label_change():
    rows = [
        ann(annotation_id=1, frame_id=10, video_id=1, frame_number=100, label_id=1),
        ann(annotation_id=2, frame_id=11, video_id=1, frame_number=101, label_id=1),
        ann(annotation_id=3, frame_id=12, video_id=1, frame_number=102, label_id=2),
        ann(annotation_id=4, frame_id=13, video_id=1, frame_number=103, label_id=2),
    ]

    sampled = sample_annotations_temporally(
        rows,
        enabled=True,
        max_frames_per_sequence=1,
        min_distance_frames=1,
        sequence_gap_frames=250,
        split_on_label_change=True,
        rare_label_aware=True,
    )

    # Same temporal area, but label changed, therefore two sequences.
    assert len(frame_ids(sampled)) == 2


def test_sampler_respects_min_distance_between_selected_frames():
    rows = [
        ann(
            annotation_id=i,
            frame_id=1000 + i,
            video_id=1,
            frame_number=100 + i,
            label_id=1,
        )
        for i in range(20)
    ]

    sampled = sample_annotations_temporally(
        rows,
        enabled=True,
        max_frames_per_sequence=5,
        min_distance_frames=5,
        sequence_gap_frames=999,
        split_on_label_change=False,
        rare_label_aware=True,
    )

    selected_numbers = sorted({row["frame"]["frame_number"] for row in sampled})

    for left, right in zip(selected_numbers, selected_numbers[1:]):
        assert right - left >= 5


def test_sampler_handles_multiple_videos_independently():
    rows = [
        ann(annotation_id=1, frame_id=10, video_id=1, frame_number=100, label_id=1),
        ann(annotation_id=2, frame_id=11, video_id=1, frame_number=101, label_id=1),
        ann(annotation_id=3, frame_id=20, video_id=2, frame_number=100, label_id=1),
        ann(annotation_id=4, frame_id=21, video_id=2, frame_number=101, label_id=1),
    ]

    sampled = sample_annotations_temporally(
        rows,
        enabled=True,
        max_frames_per_sequence=1,
        min_distance_frames=1,
        sequence_gap_frames=999,
        split_on_label_change=False,
        rare_label_aware=True,
    )

    selected_video_ids = {row["frame"]["video_id"] for row in sampled}

    assert selected_video_ids == {1, 2}
    assert len(frame_ids(sampled)) == 2


def test_sampler_prefers_multilabel_frame_when_rare_label_aware():
    rows = [
        ann(annotation_id=1, frame_id=10, video_id=1, frame_number=100, label_id=1),
        ann(annotation_id=2, frame_id=11, video_id=1, frame_number=200, label_id=1),
        ann(annotation_id=3, frame_id=11, video_id=1, frame_number=200, label_id=2),
    ]

    sampled = sample_annotations_temporally(
        rows,
        enabled=True,
        max_frames_per_sequence=1,
        min_distance_frames=1,
        sequence_gap_frames=999,
        split_on_label_change=False,
        rare_label_aware=True,
    )

    assert frame_ids(sampled) == {11}
    assert {row["label"]["id"] for row in sampled} == {1, 2}


def test_sampler_returns_original_when_frame_metadata_missing():
    rows = [
        {
            "annotation_id": 1,
            "value": True,
            "label": {"id": 1, "name": "water_jet"},
            "frame": {},
        }
    ]

    sampled = sample_annotations_temporally(
        rows,
        enabled=True,
        max_frames_per_sequence=5,
        min_distance_frames=50,
        sequence_gap_frames=250,
        split_on_label_change=True,
        rare_label_aware=True,
    )

    assert sampled == rows
