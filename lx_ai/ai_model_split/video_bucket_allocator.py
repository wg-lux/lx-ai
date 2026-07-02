# lx_ai/ai_model_split/video_bucket_allocator.py
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from lx_ai.ai_model_config.config import TrainingConfig
from lx_ai.ai_model_split.video_bucket_registry import VideoBucketRegistry
from lx_ai.utils.logging_utils import (
    decision_section,
    decision_subsection,
    kv,
    soft_line,
    subsection,
    table_header,
)


@dataclass(frozen=True)
class VideoSummary:
    video_key: str
    video_id: int
    frame_indices: List[int]
    frame_count: int
    dataset_frame_counts: Dict[int, int]
    pos_counts: List[int]
    neg_counts: List[int]
    known_counts: List[int]
    unknown_counts: List[int]


class PersistentBucketAssignmentResult(TypedDict):
    bucket_ids_per_sample: List[int]
    bucket_sizes: Dict[str, int]
    role_sizes: Dict[str, int]
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    bucket_map: Dict[str, int]
    diagnostics: Dict[str, Any]


class CandidateScoreBreakdown(TypedDict):
    bucket_id: int
    total_score: float
    frame_score: float
    video_score: float
    pos_score: float
    neg_score: float
    known_score: float
    dataset_score: float


@dataclass
class BucketStats:
    frames: int
    videos: int
    pos_counts: List[int]
    neg_counts: List[int]
    known_counts: List[int]
    dataset_frame_counts: Dict[int, int]


def _build_video_key(video_id: int) -> str:
    return f"video:{video_id}"


def _detect_condition(
    *,
    treat_unlabeled_as_negative: bool,
    all_video_summaries: Sequence[VideoSummary],
) -> str:
    """
    Returns one of:
      - 'closed_world'
      - 'partial_with_negatives'
      - 'positives_only'
    """
    if treat_unlabeled_as_negative:
        return "closed_world"

    has_any_known_negative = any(
        any(x > 0 for x in v.neg_counts) for v in all_video_summaries
    )
    if has_any_known_negative:
        return "partial_with_negatives"

    return "positives_only"


def _summarize_videos(
    *,
    video_ids: Sequence[int],
    dataset_ids_per_frame: Sequence[int],
    label_vectors: Sequence[Sequence[Optional[int]]],
    label_masks: Sequence[Sequence[int]],
) -> List[VideoSummary]:
    if not video_ids:
        raise ValueError("video_ids is empty")
    if len(video_ids) != len(label_vectors) or len(video_ids) != len(label_masks):
        raise ValueError("video_ids, label_vectors, label_masks must align")

    num_labels = len(label_vectors[0])
    grouped_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, vid in enumerate(video_ids):
        grouped_indices[int(vid)].append(idx)

    out: List[VideoSummary] = []

    for video_id in sorted(grouped_indices.keys()):
        indices = grouped_indices[video_id]

        pos_counts = [0] * num_labels
        neg_counts = [0] * num_labels
        known_counts = [0] * num_labels
        unknown_counts = [0] * num_labels
        dataset_counter: Counter[int] = Counter()

        for i in indices:
            dataset_counter[int(dataset_ids_per_frame[i])] += 1
            vec = label_vectors[i]
            mask = label_masks[i]

            for j, (x, m) in enumerate(zip(vec, mask)):
                if int(m) == 1:
                    known_counts[j] += 1
                    if x == 1:
                        pos_counts[j] += 1
                    elif x == 0:
                        neg_counts[j] += 1
                else:
                    unknown_counts[j] += 1

        out.append(
            VideoSummary(
                video_key=_build_video_key(video_id),
                video_id=int(video_id),
                frame_indices=list(indices),
                frame_count=len(indices),
                dataset_frame_counts=dict(sorted(dataset_counter.items())),
                pos_counts=pos_counts,
                neg_counts=neg_counts,
                known_counts=known_counts,
                unknown_counts=unknown_counts,
            )
        )

    return out


def _init_bucket_stats(num_buckets: int, num_labels: int) -> List[BucketStats]:
    return [
        BucketStats(
            frames=0,
            videos=0,
            pos_counts=[0] * num_labels,
            neg_counts=[0] * num_labels,
            known_counts=[0] * num_labels,
            dataset_frame_counts={},
        )
        for _ in range(num_buckets)
    ]


def _add_video_to_bucket_stats(stats: BucketStats, video: VideoSummary) -> None:
    stats.frames += int(video.frame_count)
    stats.videos += 1

    for j in range(len(video.pos_counts)):
        stats.pos_counts[j] += int(video.pos_counts[j])
        stats.neg_counts[j] += int(video.neg_counts[j])
        stats.known_counts[j] += int(video.known_counts[j])

    for ds_id, cnt in video.dataset_frame_counts.items():
        stats.dataset_frame_counts[ds_id] = stats.dataset_frame_counts.get(
            ds_id, 0
        ) + int(cnt)


def _copy_stats(stats: BucketStats) -> BucketStats:
    return BucketStats(
        frames=int(stats.frames),
        videos=int(stats.videos),
        pos_counts=list(stats.pos_counts),
        neg_counts=list(stats.neg_counts),
        known_counts=list(stats.known_counts),
        dataset_frame_counts=dict(stats.dataset_frame_counts),
    )


def _print_video_grouping_summary(
    *,
    videos: Sequence[VideoSummary],
    num_labels: int,
    label_names: Sequence[str] | None = None,
) -> None:
    subsection("VIDEO GROUPING SUMMARY")

    total_videos = len(videos)
    total_frames = sum(v.frame_count for v in videos)

    dataset_to_videos: Dict[int, int] = {}
    dataset_to_frames: Dict[int, int] = {}

    for v in videos:
        for ds_id, cnt in v.dataset_frame_counts.items():
            dataset_to_frames[ds_id] = dataset_to_frames.get(ds_id, 0) + int(cnt)
            dataset_to_videos[ds_id] = dataset_to_videos.get(ds_id, 0) + 1

    print(f"  Total videos    : {total_videos}")
    print(f"  Total frames    : {total_frames}")
    print(f"  Total datasets  : {len(dataset_to_frames)}")

    if dataset_to_frames:
        print()
        table_header("Dataset", "Videos", "Frames")
        for ds_id in sorted(dataset_to_frames.keys()):
            print(
                f"{ds_id:<12}"
                f"{dataset_to_videos.get(ds_id, 0):<12}"
                f"{dataset_to_frames[ds_id]:<12}"
            )

    print()
    table_header("Video", "Frames", "Datasets")
    for v in videos:
        datasets_str = ",".join(str(x) for x in sorted(v.dataset_frame_counts.keys()))
        print(f"{v.video_id:<12}{v.frame_count:<12}{datasets_str}")

    print()
    subsection("PER-VIDEO LABEL SUPPORT")
    for v in videos:
        print(f"  Video {v.video_id} ({v.video_key})")
        table_header("LabelIdx", "LabelName", "Pos", "Neg", "Known", "Unknown")
        for j in range(num_labels):
            label_name = (
                label_names[j]
                if label_names is not None and j < len(label_names)
                else f"label_{j}"
            )
            print(
                f"{j:<12}"
                f"{label_name:<20}"
                f"{v.pos_counts[j]:<12}"
                f"{v.neg_counts[j]:<12}"
                f"{v.known_counts[j]:<12}"
                f"{v.unknown_counts[j]:<12}"
            )
        print("-" * 80)


def _print_allocator_condition(condition: str) -> None:
    subsection("BUCKET ALLOCATION MODE")
    if condition == "closed_world":
        print("  Condition : CLOSED WORLD")
        print("  Meaning   : unknown labels are treated as negatives")
        print("  Scoring   : positives + negatives + frames + dataset spread")
    elif condition == "partial_with_negatives":
        print("  Condition : PARTIAL LABELS WITH TRUE NEGATIVES")
        print(
            "  Meaning   : unknown labels are ignored; true positives and true negatives are both used"
        )
        print(
            "  Scoring   : positives + negatives + known support + frames + dataset spread"
        )
    elif condition == "positives_only":
        print("  Condition : POSITIVES ONLY")
        print("  Meaning   : unknown labels are ignored; no known negatives available")
        print("  Scoring   : positives + frames + dataset spread")
    else:
        print(f"  Condition : {condition}")


def _print_registry_summary(
    *,
    registry_path: Path,
    old_videos: Sequence[VideoSummary],
    new_videos: Sequence[VideoSummary],
    assigned_bucket_by_video: Dict[str, int],
) -> None:
    subsection("VIDEO BUCKET REGISTRY SUMMARY")
    print(f"  Registry path         : {registry_path}")
    print(f"  Existing videos       : {len(old_videos)}")
    print(f"  New videos to assign  : {len(new_videos)}")

    if old_videos:
        print()
        table_header("Video", "Bucket", "Frames")
        for v in old_videos:
            b = assigned_bucket_by_video[v.video_key]
            print(f"{v.video_id:<12}{b:<12}{v.frame_count:<12}")


def _print_final_video_bucket_assignments(
    *,
    videos: Sequence[VideoSummary],
    assigned_bucket_by_video: Dict[str, int],
) -> None:
    subsection("FINAL VIDEO -> BUCKET ASSIGNMENTS")
    table_header("Video", "Bucket", "Frames", "Datasets")

    for v in videos:
        bucket = assigned_bucket_by_video[v.video_key]
        datasets_str = ",".join(str(x) for x in sorted(v.dataset_frame_counts.keys()))
        print(f"{v.video_id:<12}{bucket:<12}{v.frame_count:<12}{datasets_str}")


def _print_candidate_scores(
    *,
    video: VideoSummary,
    candidate_scores: Sequence[tuple[float, int]],
) -> None:
    subsection(f"NEW VIDEO ASSIGNMENT: video_id={video.video_id}")
    print(f"  Video key     : {video.video_key}")
    print(f"  Frame count   : {video.frame_count}")

    print()
    table_header("Bucket", "Score")
    for score, bucket_id in sorted(candidate_scores, key=lambda x: (x[1], x[0])):
        print(f"{bucket_id:<12}{score:<12.8f}")

    best_score, best_bucket = sorted(candidate_scores, key=lambda x: (x[0], x[1]))[0]
    print()
    print(f"  Selected bucket : {best_bucket}")
    print(f"  Selected score  : {best_score:.8f}")


def _print_new_video_decision_process(
    *,
    video: VideoSummary,
    condition: str,
    candidate_breakdowns: Sequence[CandidateScoreBreakdown],
    label_names: Sequence[str] | None = None,
) -> None:
    decision_section("NEW VIDEO BUCKET DECISION")

    datasets_str = ",".join(str(x) for x in sorted(video.dataset_frame_counts.keys()))
    kv("Video ID", video.video_id)
    kv("Video key", video.video_key)
    kv("Frames", video.frame_count)
    kv("Datasets", datasets_str)
    kv("Mode", condition)

    decision_subsection("Label support for this video")
    table_header("LabelIdx", "LabelName", "Pos", "Neg", "Known", "Unknown")
    for j in range(len(video.pos_counts)):
        label_name = (
            label_names[j]
            if label_names is not None and j < len(label_names)
            else f"label_{j}"
        )
        print(
            f"{j:<12}"
            f"{label_name:<20}"
            f"{video.pos_counts[j]:<12}"
            f"{video.neg_counts[j]:<12}"
            f"{video.known_counts[j]:<12}"
            f"{video.unknown_counts[j]:<12}"
        )

    decision_subsection("Candidate bucket scores")
    table_header(
        "Bucket",
        "Total",
        "Frames",
        "Videos",
        "Pos",
        "Neg",
        "Known",
        "Dataset",
    )
    for row in sorted(candidate_breakdowns, key=lambda x: x["bucket_id"]):
        print(
            f"{row['bucket_id']:<12}"
            f"{row['total_score']:<12.6f}"
            f"{row['frame_score']:<12.6f}"
            f"{row['video_score']:<12.6f}"
            f"{row['pos_score']:<12.6f}"
            f"{row['neg_score']:<12.6f}"
            f"{row['known_score']:<12.6f}"
            f"{row['dataset_score']:<12.6f}"
        )

    best = sorted(
        candidate_breakdowns, key=lambda x: (x["total_score"], x["bucket_id"])
    )[0]

    decision_subsection("Decision")
    kv("Selected bucket", best["bucket_id"])
    kv("Winning score", f"{best['total_score']:.6f}")
    kv(
        "Reason",
        "Minimum total score after simulating this video in every bucket",
    )
    soft_line()


def _print_bucket_balance_summary(
    *,
    bucket_stats: Sequence[BucketStats],
    label_names: Sequence[str] | None = None,
) -> None:
    subsection("BUCKET BALANCE SUMMARY")

    table_header("Bucket", "Frames", "Videos")
    for idx, st in enumerate(bucket_stats):
        print(f"{idx:<12}{st.frames:<12}{st.videos:<12}")

    if not bucket_stats:
        return

    num_labels = len(bucket_stats[0].pos_counts)

    for j in range(num_labels):
        print()
        label_name = (
            label_names[j]
            if label_names is not None and j < len(label_names)
            else f"label_{j}"
        )
        subsection(f"BUCKET LABEL BALANCE - label_index={j} ({label_name})")
        table_header("Bucket", "LabelName", "Pos", "Neg", "Known")
        for idx, st in enumerate(bucket_stats):
            print(
                f"{idx:<12}"
                f"{label_name:<20}"
                f"{st.pos_counts[j]:<12}"
                f"{st.neg_counts[j]:<12}"
                f"{st.known_counts[j]:<12}"
            )


def _label_weights_from_positive_totals(total_pos: Sequence[int]) -> List[float]:
    import math

    return [1.0 / math.sqrt(float(x) + 1.0) for x in total_pos]


def _compute_score_for_candidate_bucket(
    *,
    candidate_bucket_id: int,
    candidate_video: VideoSummary,
    current_bucket_stats: Sequence[BucketStats],
    total_frames_all_videos: int,
    total_videos_all_videos: int,
    total_pos_all_videos: Sequence[int],
    total_neg_all_videos: Sequence[int],
    total_known_all_videos: Sequence[int],
    condition: str,
) -> CandidateScoreBreakdown:
    """
    Lower score is better.
    Returns both total score and component breakdown for logging.
    """
    num_buckets = len(current_bucket_stats)
    eps = 1e-9

    simulated = [_copy_stats(x) for x in current_bucket_stats]
    _add_video_to_bucket_stats(simulated[candidate_bucket_id], candidate_video)

    target_frames = float(total_frames_all_videos) / float(num_buckets)
    target_videos = float(total_videos_all_videos) / float(num_buckets)

    if condition == "closed_world":
        w_frame = 3.0
        w_video = 0.5
        w_pos = 4.0
        w_neg = 2.5
        w_known = 0.5
        w_dataset = 1.5
    elif condition == "partial_with_negatives":
        w_frame = 3.0
        w_video = 0.5
        w_pos = 4.0
        w_neg = 3.5
        w_known = 1.5
        w_dataset = 1.5
    elif condition == "positives_only":
        w_frame = 3.0
        w_video = 0.5
        w_pos = 5.0
        w_neg = 0.0
        w_known = 0.5
        w_dataset = 1.5
    else:
        raise ValueError(f"Unknown condition={condition!r}")

    frame_score = 0.0
    video_score = 0.0
    pos_score = 0.0
    neg_score = 0.0
    known_score = 0.0
    dataset_score = 0.0

    # A. frame balance
    for st in simulated:
        frame_score += (
            w_frame * ((float(st.frames) - target_frames) / (target_frames + eps)) ** 2
        )

    # B. video count balance
    for st in simulated:
        video_score += (
            w_video * ((float(st.videos) - target_videos) / (target_videos + eps)) ** 2
        )

    # C. label balance
    label_weights = _label_weights_from_positive_totals(total_pos_all_videos)
    num_labels = len(total_pos_all_videos)

    for j in range(num_labels):
        target_pos = float(total_pos_all_videos[j]) / float(num_buckets)
        if total_pos_all_videos[j] > 0:
            for st in simulated:
                pos_score += (
                    w_pos
                    * label_weights[j]
                    * ((float(st.pos_counts[j]) - target_pos) / (target_pos + 1.0)) ** 2
                )

        if w_neg > 0.0 and total_neg_all_videos[j] > 0:
            target_neg = float(total_neg_all_videos[j]) / float(num_buckets)
            for st in simulated:
                neg_score += (
                    w_neg
                    * label_weights[j]
                    * ((float(st.neg_counts[j]) - target_neg) / (target_neg + 1.0)) ** 2
                )

        if w_known > 0.0 and total_known_all_videos[j] > 0:
            target_known = float(total_known_all_videos[j]) / float(num_buckets)
            for st in simulated:
                known_score += (
                    w_known
                    * (
                        (float(st.known_counts[j]) - target_known)
                        / (target_known + 1.0)
                    )
                    ** 2
                )

    # D. dataset concentration balance
    dataset_ids = sorted(
        {ds_id for st in simulated for ds_id in st.dataset_frame_counts.keys()}
    )
    for ds_id in dataset_ids:
        total_ds_frames = sum(st.dataset_frame_counts.get(ds_id, 0) for st in simulated)
        if total_ds_frames <= 0:
            continue
        target_ds = float(total_ds_frames) / float(num_buckets)
        for st in simulated:
            actual = float(st.dataset_frame_counts.get(ds_id, 0))
            dataset_score += w_dataset * ((actual - target_ds) / (target_ds + 1.0)) ** 2

    total_score = (
        frame_score + video_score + pos_score + neg_score + known_score + dataset_score
    )

    total_score += float(candidate_bucket_id) * 1e-12

    return {
        "bucket_id": int(candidate_bucket_id),
        "total_score": float(total_score),
        "frame_score": float(frame_score),
        "video_score": float(video_score),
        "pos_score": float(pos_score),
        "neg_score": float(neg_score),
        "known_score": float(known_score),
        "dataset_score": float(dataset_score),
    }


def _video_hardness(
    *,
    video: VideoSummary,
    total_pos_all_videos: Sequence[int],
    total_neg_all_videos: Sequence[int],
) -> float:
    """
    Larger means harder to place.
    Prioritize:
      - bigger videos
      - rare-label-heavy videos
      - negative-heavy videos later when they exist
    """
    import math

    rare_mass = 0.0
    for p, total_p in zip(video.pos_counts, total_pos_all_videos):
        if p > 0:
            rare_mass += float(p) / math.sqrt(float(total_p) + 1.0)

    neg_mass = 0.0
    for n, total_n in zip(video.neg_counts, total_neg_all_videos):
        if n > 0:
            neg_mass += float(n) / math.sqrt(float(total_n) + 1.0)

    return (
        3.0 * float(video.frame_count) + 10.0 * float(rare_mass) + 5.0 * float(neg_mass)
    )


def assign_buckets_with_persistent_video_registry(
    *,
    config: TrainingConfig,
    video_ids: Sequence[int],
    dataset_ids_per_frame: Sequence[int],
    label_vectors: Sequence[Sequence[Optional[int]]],
    label_masks: Sequence[Sequence[int]],
    label_names: Sequence[str] | None = None,
) -> PersistentBucketAssignmentResult:
    """
    Main entry point:
      - existing video assignments are reused from registry
      - new videos are assigned greedily with multi-objective scoring
      - returns per-sample bucket ids and split indices compatible with existing pipeline
    """
    num_samples = len(video_ids)
    if not (
        len(video_ids)
        == len(dataset_ids_per_frame)
        == len(label_vectors)
        == len(label_masks)
    ):
        raise ValueError(
            "video_ids, dataset_ids_per_frame, label_vectors, label_masks must align"
        )
    if num_samples == 0:
        raise ValueError("No samples to assign")

    num_buckets = int(config.bucket_policy.num_buckets)
    registry_path = (
        Path(config.training_root) / "bucket_registry" / "video_bucket_registry.json"
    )
    registry = VideoBucketRegistry.load(path=registry_path, num_buckets=num_buckets)

    all_videos = _summarize_videos(
        video_ids=video_ids,
        dataset_ids_per_frame=dataset_ids_per_frame,
        label_vectors=label_vectors,
        label_masks=label_masks,
    )
    num_labels = len(all_videos[0].pos_counts)

    """_print_video_grouping_summary(
        videos=all_videos,
        num_labels=num_labels,
        label_names=label_names,
    )"""

    total_frames_all = sum(v.frame_count for v in all_videos)
    total_videos_all = len(all_videos)
    total_pos_all = [0] * num_labels
    total_neg_all = [0] * num_labels
    total_known_all = [0] * num_labels

    for v in all_videos:
        for j in range(num_labels):
            total_pos_all[j] += int(v.pos_counts[j])
            total_neg_all[j] += int(v.neg_counts[j])
            total_known_all[j] += int(v.known_counts[j])

    condition = _detect_condition(
        treat_unlabeled_as_negative=bool(config.treat_unlabeled_as_negative),
        all_video_summaries=all_videos,
    )

    # _print_allocator_condition(condition)

    diagnostics: Dict[str, Any] = {
        "condition": condition,
        "registry_path": str(registry_path),
        "video_grouping": {
            "total_videos": len(all_videos),
            "total_frames": sum(v.frame_count for v in all_videos),
            "total_datasets": len(
                {ds_id for v in all_videos for ds_id in v.dataset_frame_counts.keys()}
            ),
        },
        "existing_videos": [],
        "new_videos": [],
        "new_video_decisions": [],
        "final_assignments": [],
        "bucket_balance": [],
    }

    bucket_stats = _init_bucket_stats(num_buckets=num_buckets, num_labels=num_labels)
    assigned_bucket_by_video: Dict[str, int] = {}

    old_videos: List[VideoSummary] = []
    new_videos: List[VideoSummary] = []

    for v in all_videos:
        existing = registry.get(v.video_key)
        if existing is None:
            new_videos.append(v)
        else:
            assigned_bucket_by_video[v.video_key] = int(existing)
            old_videos.append(v)

    diagnostics["existing_videos"] = [
        {
            "video_id": v.video_id,
            "video_key": v.video_key,
            "frames": v.frame_count,
            "bucket": assigned_bucket_by_video[v.video_key],
            "datasets": sorted(v.dataset_frame_counts.keys()),
        }
        for v in old_videos
    ]

    diagnostics["new_videos"] = [
        {
            "video_id": v.video_id,
            "video_key": v.video_key,
            "frames": v.frame_count,
            "datasets": sorted(v.dataset_frame_counts.keys()),
            "pos_counts": list(v.pos_counts),
            "neg_counts": list(v.neg_counts),
            "known_counts": list(v.known_counts),
            "unknown_counts": list(v.unknown_counts),
        }
        for v in new_videos
    ]

    """_print_registry_summary(
        registry_path=registry_path,
        old_videos=old_videos,
        new_videos=new_videos,
        assigned_bucket_by_video=assigned_bucket_by_video,
    )"""

    # Seed bucket stats with already-frozen assignments
    for v in old_videos:
        b = assigned_bucket_by_video[v.video_key]
        _add_video_to_bucket_stats(bucket_stats[b], v)

    # Hardest-first for new videos
    new_videos_sorted = sorted(
        new_videos,
        key=lambda x: (
            -_video_hardness(
                video=x,
                total_pos_all_videos=total_pos_all,
                total_neg_all_videos=total_neg_all,
            ),
            x.video_key,
        ),
    )

    #
    """for v in new_videos_sorted:
        candidate_scores: List[tuple[float, int]] = []
        for b in range(num_buckets):
            s = _compute_score_for_candidate_bucket(
                candidate_bucket_id=b,
                candidate_video=v,
                current_bucket_stats=bucket_stats,
                total_frames_all_videos=total_frames_all,
                total_videos_all_videos=total_videos_all,
                total_pos_all_videos=total_pos_all,
                total_neg_all_videos=total_neg_all,
                total_known_all_videos=total_known_all,
                condition=condition,
            )
            candidate_scores.append((s, b))



        _print_candidate_scores(
            video=v,
            candidate_scores=candidate_scores,
        )

        candidate_scores.sort(key=lambda x: (x[0], x[1]))
        best_bucket = int(candidate_scores[0][1])

        assigned_bucket_by_video[v.video_key] = best_bucket
        registry.set(v.video_key, best_bucket)
        _add_video_to_bucket_stats(bucket_stats[best_bucket], v)"""
    #

    for v in new_videos_sorted:
        candidate_breakdowns: List[CandidateScoreBreakdown] = []
        for b in range(num_buckets):
            breakdown = _compute_score_for_candidate_bucket(
                candidate_bucket_id=b,
                candidate_video=v,
                current_bucket_stats=bucket_stats,
                total_frames_all_videos=total_frames_all,
                total_videos_all_videos=total_videos_all,
                total_pos_all_videos=total_pos_all,
                total_neg_all_videos=total_neg_all,
                total_known_all_videos=total_known_all,
                condition=condition,
            )
            candidate_breakdowns.append(breakdown)

        """_print_new_video_decision_process(
            video=v,
            condition=condition,
            candidate_breakdowns=candidate_breakdowns,
            label_names=label_names,
        )"""

        diagnostics["new_video_decisions"].append(
            {
                "video_id": v.video_id,
                "video_key": v.video_key,
                "frames": v.frame_count,
                "datasets": sorted(v.dataset_frame_counts.keys()),
                "condition": condition,
                "pos_counts": list(v.pos_counts),
                "neg_counts": list(v.neg_counts),
                "known_counts": list(v.known_counts),
                "unknown_counts": list(v.unknown_counts),
                "candidate_breakdowns": candidate_breakdowns,
            }
        )

        best = sorted(
            candidate_breakdowns, key=lambda x: (x["total_score"], x["bucket_id"])
        )[0]
        best_bucket = int(best["bucket_id"])

        assigned_bucket_by_video[v.video_key] = best_bucket
        registry.set(v.video_key, best_bucket)
        _add_video_to_bucket_stats(bucket_stats[best_bucket], v)

    registry.save()

    diagnostics["final_assignments"] = [
        {
            "video_id": v.video_id,
            "video_key": v.video_key,
            "frames": v.frame_count,
            "bucket": assigned_bucket_by_video[v.video_key],
            "datasets": sorted(v.dataset_frame_counts.keys()),
        }
        for v in all_videos
    ]

    diagnostics["bucket_balance"] = [
        {
            "bucket_id": idx,
            "frames": st.frames,
            "videos": st.videos,
            "pos_counts": list(st.pos_counts),
            "neg_counts": list(st.neg_counts),
            "known_counts": list(st.known_counts),
        }
        for idx, st in enumerate(bucket_stats)
    ]

    """_print_final_video_bucket_assignments(
        videos=all_videos,
        assigned_bucket_by_video=assigned_bucket_by_video,
    )"""

    """_print_bucket_balance_summary(
        bucket_stats=bucket_stats,
        label_names=label_names,
    )"""

    bucket_ids_per_sample: List[int] = []
    bucket_map: Dict[str, int] = dict(sorted(assigned_bucket_by_video.items()))

    for vid in video_ids:
        video_key = _build_video_key(int(vid))
        b = assigned_bucket_by_video.get(video_key)
        if b is None:
            raise RuntimeError(f"Missing bucket assignment for video_key={video_key}")
        bucket_ids_per_sample.append(int(b))

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    val_buckets = set(int(x) for x in config.bucket_policy.validation_buckets)
    test_buckets = set(int(x) for x in config.bucket_policy.test_buckets)

    for i, b in enumerate(bucket_ids_per_sample):
        if b in val_buckets:
            val_idx.append(i)
        elif b in test_buckets:
            test_idx.append(i)
        else:
            train_idx.append(i)

    bucket_counts = Counter(bucket_ids_per_sample)
    bucket_sizes = {str(k): int(v) for k, v in sorted(bucket_counts.items())}
    role_sizes = {
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
    }

    return {
        "bucket_ids_per_sample": bucket_ids_per_sample,
        "bucket_sizes": bucket_sizes,
        "role_sizes": role_sizes,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "bucket_map": bucket_map,
        "diagnostics": diagnostics,
    }
