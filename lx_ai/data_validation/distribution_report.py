# lx_ai/data_validation/distribution_report.py
from __future__ import annotations
from lx_ai.utils.logging_utils import section, subsection, table_header
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, cast

import torch


# ----------------------------
# Public output shapes
# ----------------------------
class LabelStats(TypedDict):
    label_index: int
    label_id: Optional[int]
    label_name: str
    positives: int
    negatives: int
    known: int
    unknown: int
    pos_rate: Optional[float]          # positives / known
    imbalance_ratio: Optional[float]   # max(pos,neg)/min(pos,neg) among known (None if cannot define)


class SplitLabelSummary(TypedDict):
    split_name: str
    num_samples: int
    num_labels: int
    labels: List[LabelStats]


class SplitComparison(TypedDict):
    metric: str
    train_vs_val: Optional[float]
    train_vs_test: Optional[float]
    val_vs_test: Optional[float]


class ExamSummary(TypedDict):
    split_name: str
    unique_exams: int
    frames: int
    mean_frames_per_exam: float
    max_frames_per_exam: int


class DataValidationReport(TypedDict):
    version: str
    dataset_uuid: str
    num_labels: int
    splits: List[SplitLabelSummary]
    label_distribution_similarity: List[SplitComparison]
    examinations: List[ExamSummary]


# ----------------------------
# Helpers
# ----------------------------
_EPS = 1e-12


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _label_name(lbl: Any) -> str:
    n = _get(lbl, "name", None)
    if isinstance(n, str) and n.strip():
        return n.strip()
    return "<unnamed>"


def _label_id(lbl: Any) -> Optional[int]:
    i = _get(lbl, "id", None)
    return i if isinstance(i, int) else None


def _safe_div(a: float, b: float) -> Optional[float]:
    if b <= 0:
        return None
    return a / b


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Jensen-Shannon divergence between two discrete distributions.
    Returns value in [0, ln(2)] (natural log).
    """
    p = p.double().clamp_min(_EPS)
    q = q.double().clamp_min(_EPS)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return float(0.5 * (kl_pm + kl_qm))


def _compute_split_label_stats(
    *,
    label_vectors: Sequence[Sequence[Optional[int]]],
    label_masks: Sequence[Sequence[int]],
    labels_any: Sequence[Any],
    indices: Sequence[int],
    split_name: str,
) -> SplitLabelSummary:
    # Convert to tensors for fast aggregation
    # label_vectors: values may contain None, so we map None->0 for tensor math
    vec = []
    msk = []

    for i in indices:
        row_v = label_vectors[i]
        row_m = label_masks[i]
        vec.append([0 if x is None else int(x) for x in row_v])
        msk.append([int(x) for x in row_m])

    if not vec:
        return {
            "split_name": split_name,
            "num_samples": 0,
            "num_labels": len(labels_any),
            "labels": [],
        }

    v_t = torch.tensor(vec, dtype=torch.int64)     # [N,C]
    m_t = torch.tensor(msk, dtype=torch.int64)     # [N,C]

    known = (m_t == 1)
    unknown = (m_t == 0)

    positives = ((v_t == 1) & known).sum(dim=0)  # [C]
    negatives = ((v_t == 0) & known).sum(dim=0)  # [C]
    known_counts = known.sum(dim=0)              # [C]
    unknown_counts = unknown.sum(dim=0)          # [C]

    out: List[LabelStats] = []
    c = int(v_t.shape[1])

    for j in range(c):
        pos = int(positives[j].item())
        neg = int(negatives[j].item())
        kn = int(known_counts[j].item())
        un = int(unknown_counts[j].item())

        pos_rate = _safe_div(float(pos), float(kn)) if kn > 0 else None

        # imbalance ratio: max(pos,neg)/min(pos,neg) among known
        if pos > 0 and neg > 0:
            imbalance = float(max(pos, neg) / min(pos, neg))
        else:
            imbalance = None

        lbl = labels_any[j] if j < len(labels_any) else {}
        out.append(
            {
                "label_index": j,
                "label_id": _label_id(lbl),
                "label_name": _label_name(lbl),
                "positives": pos,
                "negatives": neg,
                "known": kn,
                "unknown": un,
                "pos_rate": pos_rate,
                "imbalance_ratio": imbalance,
            }
        )

    return {
        "split_name": split_name,
        "num_samples": int(len(indices)),
        "num_labels": int(c),
        "labels": out,
    }


def _split_similarity_from_pos_rates(
    a: SplitLabelSummary, b: SplitLabelSummary
) -> Optional[float]:
    """
    Compare label distribution similarity using Jensen-Shannon divergence
    over per-label positive rates (smoothed).

    - Small value => similar
    - Larger value => different
    """
    if not a["labels"] or not b["labels"]:
        return None

    # build vectors of positive rates *known counts as weights
    # For stability, we use "positives + 1" over "known + 2" (Laplace smoothing)
    pa = []
    pb = []

    for la, lb in zip(a["labels"], b["labels"]):
        ka = la["known"]
        kb = lb["known"]
        posa = la["positives"]
        posb = lb["positives"]

        # Use positive counts distribution across labels (more standard than rate vector)
        pa.append(float(posa) + 1.0)
        pb.append(float(posb) + 1.0)

    p = torch.tensor(pa, dtype=torch.float64)
    q = torch.tensor(pb, dtype=torch.float64)
    return _js_divergence(p, q)


def _exam_summary(
    *,
    old_examination_ids: Sequence[Optional[int]],
    indices: Sequence[int],
    split_name: str,
) -> ExamSummary:
    # group by exam; if None -> treat each sample as its own group (fallback)
    counts: Dict[str, int] = {}

    for i in indices:
        eid = old_examination_ids[i]
        key = f"exam:{eid}" if eid is not None else f"frame:{i}"
        counts[key] = counts.get(key, 0) + 1

    if not counts:
        return {
            "split_name": split_name,
            "unique_exams": 0,
            "frames": 0,
            "mean_frames_per_exam": 0.0,
            "max_frames_per_exam": 0,
        }

    total_frames = int(len(indices))
    unique_exams = int(len(counts))
    mean_frames = float(total_frames / max(unique_exams, 1))
    max_frames = int(max(counts.values()))

    return {
        "split_name": split_name,
        "unique_exams": unique_exams,
        "frames": total_frames,
        "mean_frames_per_exam": mean_frames,
        "max_frames_per_exam": max_frames,
    }


def write_data_validation_report(
    *,
    out_dir: Path,
    dataset_uuid: str,
    labels_any: Sequence[Any],
    label_vectors: Sequence[Sequence[Optional[int]]],
    label_masks: Sequence[Sequence[int]],
    frame_ids: Sequence[int],
    old_examination_ids: Sequence[Optional[int]],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
) -> Tuple[Path, Path, Path]:
    """
    Writes:
      1) data_validation_report.json
      2) label_stats.csv
      3) exam_stats.csv

    Returns paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sum = _compute_split_label_stats(
        label_vectors=label_vectors,
        label_masks=label_masks,
        labels_any=labels_any,
        indices=train_indices,
        split_name="train",
    )
    val_sum = _compute_split_label_stats(
        label_vectors=label_vectors,
        label_masks=label_masks,
        labels_any=labels_any,
        indices=val_indices,
        split_name="val",
    )
    test_sum = _compute_split_label_stats(
        label_vectors=label_vectors,
        label_masks=label_masks,
        labels_any=labels_any,
        indices=test_indices,
        split_name="test",
    )

    sim_tv = _split_similarity_from_pos_rates(train_sum, val_sum)
    sim_tt = _split_similarity_from_pos_rates(train_sum, test_sum)
    sim_vt = _split_similarity_from_pos_rates(val_sum, test_sum)

    comparisons: List[SplitComparison] = [
        {
            "metric": "JS_divergence_over_positive_counts_per_label",
            "train_vs_val": sim_tv,
            "train_vs_test": sim_tt,
            "val_vs_test": sim_vt,
        }
    ]

    exam_train = _exam_summary(old_examination_ids=old_examination_ids, indices=train_indices, split_name="train")
    exam_val = _exam_summary(old_examination_ids=old_examination_ids, indices=val_indices, split_name="val")
    exam_test = _exam_summary(old_examination_ids=old_examination_ids, indices=test_indices, split_name="test")

    report: DataValidationReport = {
        "version": "1.0",
        "dataset_uuid": dataset_uuid,
        "num_labels": int(len(labels_any)),
        "splits": [train_sum, val_sum, test_sum],
        "label_distribution_similarity": comparisons,
        "examinations": [exam_train, exam_val, exam_test],
    }

    # --- write JSON ---
    json_path = out_dir / "data_validation_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- write label CSV (one row per label per split) ---
    label_csv = out_dir / "label_stats.csv"
    with label_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "split",
            "label_index",
            "label_id",
            "label_name",
            "positives",
            "negatives",
            "known",
            "unknown",
            "pos_rate",
            "imbalance_ratio",
        ])
        for split_sum in (train_sum, val_sum, test_sum):
            for ls in split_sum["labels"]:
                w.writerow([
                    split_sum["split_name"],
                    ls["label_index"],
                    ls["label_id"],
                    ls["label_name"],
                    ls["positives"],
                    ls["negatives"],
                    ls["known"],
                    ls["unknown"],
                    ls["pos_rate"],
                    ls["imbalance_ratio"],
                ])

    # --- write exam CSV ---
    exam_csv = out_dir / "exam_stats.csv"
    with exam_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "split",
            "unique_exams",
            "frames",
            "mean_frames_per_exam",
            "max_frames_per_exam",
        ])
        for es in (exam_train, exam_val, exam_test):
            w.writerow([
                es["split_name"],
                es["unique_exams"],
                es["frames"],
                f"{es['mean_frames_per_exam']:.4f}",
                es["max_frames_per_exam"],
            ])

    return json_path, label_csv, exam_csv

def print_data_validation_report_to_console(report: DataValidationReport) -> None:
    section("DATA VALIDATION REPORT", icon="📊")

    # ---------------------------------------------------------
    # LABEL DISTRIBUTION PER SPLIT
    # ---------------------------------------------------------
    for split in report["splits"]:
        subsection(f"LABEL DISTRIBUTION - {split['split_name'].upper()}")

        if not split["labels"]:
            print("  No samples in this split.")
            continue

        table_header(
            "Label",
            "Pos",
            "Neg",
            "Known",
            "Unknown",
            "PosRate",
            "Imbal"
        )

        for ls in split["labels"]:
            pos_rate = (
                f"{ls['pos_rate']:.3f}"
                if ls["pos_rate"] is not None
                else "N/A"
            )
            imbalance = (
                f"{ls['imbalance_ratio']:.2f}"
                if ls["imbalance_ratio"] is not None
                else "N/A"
            )

            print(
                f"{ls['label_name'][:12]:<12} "
                f"{ls['positives']:<10} "
                f"{ls['negatives']:<10} "
                f"{ls['known']:<10} "
                f"{ls['unknown']:<10} "
                f"{pos_rate:<10} "
                f"{imbalance:<10}"
            )

    # ---------------------------------------------------------
    # SPLIT SIMILARITY
    # ---------------------------------------------------------
    subsection("SPLIT DISTRIBUTION SIMILARITY")

    table_header("Metric", "Train-Val", "Train-Test", "Val-Test")

    for comp in report["label_distribution_similarity"]:
        tv = (
            f"{comp['train_vs_val']:.6f}"
            if comp["train_vs_val"] is not None
            else "N/A"
        )
        tt = (
            f"{comp['train_vs_test']:.6f}"
            if comp["train_vs_test"] is not None
            else "N/A"
        )
        vt = (
            f"{comp['val_vs_test']:.6f}"
            if comp["val_vs_test"] is not None
            else "N/A"
        )

        print(
            f"{comp['metric'][:20]:<20} "
            f"{tv:<15} "
            f"{tt:<15} "
            f"{vt:<15}"
        )

    # ---------------------------------------------------------
    # EXAM DISTRIBUTION
    # ---------------------------------------------------------
    subsection("EXAMINATION DISTRIBUTION")

    table_header(
        "Split",
        "UniqueExams",
        "Frames",
        "MeanFrames",
        "MaxFrames"
    )

    for ex in report["examinations"]:
        print(
            f"{ex['split_name']:<12} "
            f"{ex['unique_exams']:<12} "
            f"{ex['frames']:<12} "
            f"{ex['mean_frames_per_exam']:<12.2f} "
            f"{ex['max_frames_per_exam']:<12}"
)