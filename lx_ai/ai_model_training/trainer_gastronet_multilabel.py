# lx_ai/ai_model_training/trainer_gastronet_multilabel.py
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, cast

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from lx_ai.utils.bucket_utils import split_data_into_buckets
from lx_ai.utils.data_accumulation import load_datasets, merge_datasets

from lx_ai.ai_model_config.config import TrainingConfig
from lx_ai.ai_model_dataset.dataset import EndoMultiLabelDataset, MultiLabelDatasetSpec
from lx_ai.ai_model.losses import compute_class_weights, focal_loss_with_mask
from lx_ai.ai_model_matrics.metrics import MetricsResult, compute_metrics
from lx_ai.ai_model.model_backbones import create_multilabel_model
from lx_ai.utils.data_loader_for_model_input import build_dataset_for_training

from lx_ai.utils.logging_utils import table_header, subsection
# -----------------------------------------------------------------------------
# Typed shapes (Pylance clarity)
# -----------------------------------------------------------------------------
class TrainResult(TypedDict):
    model_path: str
    meta_path: str
    history: Dict[str, Any]


def _is_mapping(obj: Any) -> bool:
    return isinstance(obj, dict)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if _is_mapping(obj):
        return cast(dict[str, Any], obj).get(key, default)
    return getattr(obj, key, default)


def _label_name(lbl: Any) -> str:
    n = _get(lbl, "name", None)
    if isinstance(n, str) and n.strip():
        return n.strip()
    return "<unnamed>"


def _label_has_version(lbl: Any, version: int) -> bool:
    versions = _get(lbl, "labelset_versions", None)
    if isinstance(versions, list) and all(isinstance(x, int) for x in versions):
        return version in cast(List[int], versions)

    label_sets = _get(lbl, "label_sets", None)
    if isinstance(label_sets, list):
        for ls in cast(List[Any], label_sets):
            v = _get(ls, "version", None)
            if isinstance(v, int) and v == version:
                return True

    return False


def filter_labels_by_labelset_version(
    labels: Sequence[Any],
    label_vectors: Sequence[Sequence[Optional[int]]],
    label_masks: Sequence[Sequence[int]],
    target_version: int,
    *,
    labelset: Optional[Any] = None,
) -> Tuple[
    List[List[Optional[int]]],
    List[List[int]],
    List[Any],
    List[int],
]:
    if labelset is not None:
        ls_version = _get(labelset, "version", None)
        if isinstance(ls_version, int) and ls_version == target_version:
            kept = list(range(len(labels)))
            return (
                [list(v) for v in label_vectors],
                [list(m) for m in label_masks],
                list(labels),
                kept,
            )

    kept_indices: List[int] = []
    for idx, lbl in enumerate(labels):
        if _label_has_version(lbl, target_version):
            kept_indices.append(idx)

    if not kept_indices:
        raise ValueError(
            f"No labels match labelset_version_to_train={target_version}. "
            "Either adjust config or provide label membership metadata."
        )

    filtered_vectors: List[List[Optional[int]]] = []
    filtered_masks: List[List[int]] = []

    for vec, mask in zip(label_vectors, label_masks):
        filtered_vectors.append([vec[j] for j in kept_indices])
        filtered_masks.append([mask[j] for j in kept_indices])

    filtered_labels = [labels[j] for j in kept_indices]
    return filtered_vectors, filtered_masks, filtered_labels, kept_indices


def groupwise_split_indices_by_examination(
    frame_ids: Sequence[int],
    old_examination_ids: Sequence[Optional[int]],
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    if len(frame_ids) != len(old_examination_ids):
        raise ValueError("frame_ids and old_examination_ids must have the same length")

    groups: Dict[object, List[int]] = {}
    for idx, (fid, exam_id) in enumerate(zip(frame_ids, old_examination_ids)):
        key: object = exam_id if exam_id is not None else f"no_exam_{fid}"
        groups.setdefault(key, []).append(idx)

    group_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_test = int(round(test_split * n_groups))
    n_val = int(round(val_split * n_groups))
    n_train = n_groups - n_val - n_test

    train_g = group_ids[:n_train]
    val_g = group_ids[n_train : n_train + n_val]
    test_g = group_ids[n_train + n_val :]

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for gid in train_g:
        train_idx.extend(groups[gid])
    for gid in val_g:
        val_idx.extend(groups[gid])
    for gid in test_g:
        test_idx.extend(groups[gid])

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    subsection("DATASET SPLIT (GROUP-WISE BY old_examination_id)")

    table_header("Split", "Groups", "Percentage")
    
    total = n_groups
    train_n = len(train_g)
    val_n = len(val_g)
    test_n = len(test_g)
    
    def _pct(x: int) -> float:
        return 100.0 * x / total if total > 0 else 0.0
    
    print(f"{'Train':<12} {train_n:<10d} {_pct(train_n):>6.1f} %")
    print(f"{'Validation':<12} {val_n:<10d} {_pct(val_n):>6.1f} %")
    print(f"{'Test':<12} {test_n:<10d} {_pct(test_n):>6.1f} %")
    print("-" * 80)
    print(f"{'Total':<12} {total:<10d} {100.0:>6.1f} %")

    return train_idx, val_idx, test_idx


def _tensor_row_as_floats(x: torch.Tensor, max_items: int = 12) -> List[float]:
    """
    Pylance-friendly conversion for debug printing (avoids Tensor.tolist() Unknown warnings).
    """
    arr = x.detach().to("cpu").flatten().numpy()
    out = [float(v) for v in cast(np.ndarray, arr)[:max_items]]
    return out

def train_gastronet_multilabel(config: TrainingConfig) -> TrainResult:
    #####
        # Load datasets based on dataset_ids
    datasets = load_datasets(config.dataset_ids)
    
    # Merge datasets into one
    merged_data = merge_datasets(datasets)
    
    # Perform bucket assignment based on unique identifiers (exam_id, frame_id)
    buckets = split_data_into_buckets(merged_data, num_buckets=config.num_buckets)
    
    # Divide the data into training, validation, and test sets based on bucket roles
    train_data = []
    val_data = []
    test_data = []
    for bucket_id, samples in buckets.items():
        if bucket_id in config.train_bucket_ids:
            train_data.extend(samples)
        elif bucket_id in config.val_bucket_ids:
            val_data.extend(samples)
        elif bucket_id in config.test_bucket_ids:
            test_data.extend(samples)
    #####
    data = build_dataset_for_training(config=config)
    '''data = build_dataset_for_training(
        dataset=None,
        labelset=None,
        assume_missing_is_negative=config.treat_unlabeled_as_negative,
    )'''

    #image_paths: List[str] = data["image_paths"]
    
    image_paths_str: List[str] = cast(List[str], data["image_paths"])
    image_paths: List[Path] = [Path(p) for p in image_paths_str]

    
    label_vectors: List[List[Optional[int]]] = data["label_vectors"]
    label_masks: List[List[int]] = data["label_masks"]
    labels_any: List[Any] = cast(List[Any], data["labels"])
    labelset_any: Any = data["labelset"]
    frame_ids: List[int] = data.get("frame_ids", list(range(len(image_paths))))
    old_exam_ids: List[Optional[int]] = data.get("old_examination_ids", [None] * len(image_paths))

    subsection("DATASET SUMMARY")
    print(f"  Dataset UUID        : {config.dataset_uuid}")
    print(f"  Samples (frames)    : {len(image_paths)}")
    print(f"  Labels (raw)        : {len(labels_any)}")

    (
        label_vectors,
        label_masks,
        labels_any,
        kept_indices,
    ) = filter_labels_by_labelset_version(
        labels=labels_any,
        label_vectors=label_vectors,
        label_masks=label_masks,
        target_version=config.labelset_version_to_train,
        labelset=labelset_any,
    )

    subsection("LABEL SPACE (AFTER FILTERING)")
    print(f"  Total labels kept: {len(labels_any)}\n")
    
    table_header("New idx", "Label ID", "Label name")
    
    for new_idx, lbl in enumerate(labels_any):
        lbl_id = _get(lbl, "id", None)
        name = _label_name(lbl)
        print(f"{new_idx:<10} {str(lbl_id):<10} {name}")



    # Apply unlabeled semantics
    if config.treat_unlabeled_as_negative:
        for i in range(len(label_vectors)):
            vec = label_vectors[i]
            new_vec: List[Optional[int]] = []
            new_mask: List[int] = []
            for x in vec:
                if x is None:
                    new_vec.append(0)
                    new_mask.append(1)
                else:
                    new_vec.append(int(x))
                    new_mask.append(1)
            label_vectors[i] = new_vec
            label_masks[i] = new_mask
    else:
        cleaned_vectors: List[List[Optional[int]]] = []
        cleaned_masks: List[List[int]] = []
        for vec, mask in zip(label_vectors, label_masks):
            v2: List[Optional[int]] = []
            m2: List[int] = []
            for x, ms in zip(vec, mask):
                if x is None:
                    v2.append(0)
                    m2.append(0)
                else:
                    v2.append(int(x))
                    m2.append(int(ms))
            cleaned_vectors.append(v2)
            cleaned_masks.append(m2)
        label_vectors = cleaned_vectors
        label_masks = cleaned_masks

    # Split
    '''train_indices, val_indices, test_indices = groupwise_split_indices_by_examination(
        frame_ids=frame_ids,
        old_examination_ids=old_exam_ids,
        val_split=config.val_split,
        test_split=config.test_split,
        seed=config.random_seed,
    )'''
    # Replace `groupwise_split_indices_by_examination` with `split_data_into_buckets`
    train_indices, val_indices, test_indices = split_data_into_buckets(merged_data, num_buckets=config.num_buckets)


    # Build specs (Pylance OK because MultiLabelDatasetSpec.image_paths is Sequence[Path])
    full_spec = MultiLabelDatasetSpec(
        image_paths=image_paths,
        label_vectors=label_vectors,
        label_masks=label_masks,
        image_size=224,
    )
    full_ds = EndoMultiLabelDataset(full_spec)

    def _subset_spec(spec: MultiLabelDatasetSpec, indices: List[int]) -> MultiLabelDatasetSpec:
        ip = [str(spec.image_paths[i]) for i in indices]
        lv = [spec.label_vectors[i] for i in indices]
        lm = [spec.label_masks[i] for i in indices]
        return MultiLabelDatasetSpec(
            image_paths=ip,
            label_vectors=lv,
            label_masks=lm,
            image_size=spec.image_size,
        )


    #train_ds = EndoMultiLabelDataset(_subset_spec(full_spec, train_indices))
    #val_ds = EndoMultiLabelDataset(_subset_spec(full_spec, val_indices))
    #test_ds = EndoMultiLabelDataset(_subset_spec(full_spec, test_indices))
     
    train_ds = EndoMultiLabelDataset(_subset_spec(full_spec, train_indices))

    val_ds = None
    if len(val_indices) > 0:
        val_ds = EndoMultiLabelDataset(
            _subset_spec(full_spec, val_indices)
        )
    
    test_ds = None
    if len(test_indices) > 0:
        test_ds = EndoMultiLabelDataset(
            _subset_spec(full_spec, test_indices)
        )







    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    #test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    # Device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    # Model
    model = create_multilabel_model(
        backbone_name=config.backbone_name,
        num_labels=len(labels_any),
        backbone_checkpoint=config.backbone_checkpoint,
        freeze_backbone=config.freeze_backbone,
    )
    model.to(device)

    # Class weights
    class_weights = compute_class_weights(full_ds.labels, full_ds.masks).to(device)
    subsection("CLASS WEIGHTS")
    print("  First 8 weights:", _tensor_row_as_floats(class_weights, max_items=8))


    # Optimizer + scheduler
    head_params = list(model.classifier.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": config.lr_head},
            {"params": backbone_params, "lr": config.lr_backbone},
        ]
    )

    base_lrs = [float(config.lr_head), float(config.lr_backbone)]

    scheduler: Optional[CosineAnnealingLR]
    warmup_epochs: int

    if config.use_scheduler:
        warmup_epochs = max(config.warmup_epochs, 0)
        t_max = max(config.num_epochs - warmup_epochs, 1)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=config.min_lr)
        print(f"[LR] warmup_epochs={warmup_epochs}, T_max={t_max}, min_lr={config.min_lr}")
    else:
        scheduler = None
        warmup_epochs = 0
        print("[LR] No scheduler.")

    history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "test_loss": None}

    # Debug first batch
    imgs_dbg, y_dbg, m_dbg = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        logits_dbg = model(imgs_dbg.to(device))
        probs_dbg = torch.sigmoid(logits_dbg)

    for epoch in range(1, config.num_epochs + 1):
        if scheduler is not None:
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                warmup_factor = epoch / float(warmup_epochs)
                for i, pg in enumerate(optimizer.param_groups):
                    pg["lr"] = base_lrs[i] * warmup_factor
            else:
                scheduler.step()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for imgs, y, m in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = focal_loss_with_mask(
                logits=logits,
                targets=y,
                masks=m,
                class_weights=class_weights,
                alpha=config.alpha_focal,
                gamma=config.gamma_focal,
            )
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        history["train_loss"].append(train_loss)

        # Val
        '''if val_loader is None:

            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            val_logits: List[torch.Tensor] = []
            val_targets: List[torch.Tensor] = []
            val_masks: List[torch.Tensor] = []
    
            with torch.no_grad():
                for imgs, y, m in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    m = m.to(device, non_blocking=True)
    
                    logits = model(imgs)
                    loss = focal_loss_with_mask(
                        logits=logits,
                        targets=y,
                        masks=m,
                        class_weights=class_weights,
                        alpha=config.alpha_focal,
                        gamma=config.gamma_focal,
                    )
                    val_loss_sum += float(loss.item())
                    val_batches += 1
                    val_logits.append(logits)
                    val_targets.append(y)
                    val_masks.append(m)

        val_loss = val_loss_sum / max(val_batches, 1)
        history["val_loss"].append(val_loss)

        all_val_logits = torch.cat(val_logits, dim=0)
        all_val_targets = torch.cat(val_targets, dim=0)
        all_val_masks = torch.cat(val_masks, dim=0)

        val_metrics: MetricsResult = compute_metrics(
            logits=all_val_logits, targets=all_val_targets, masks=all_val_masks, threshold=0.5
        )'''

        # -------------------------
        # Validation
        # -------------------------
        val_metrics = None
        
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            val_logits: List[torch.Tensor] = []
            val_targets: List[torch.Tensor] = []
            val_masks: List[torch.Tensor] = []
        
            with torch.no_grad():
                for imgs, y, m in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    m = m.to(device, non_blocking=True)
        
                    logits = model(imgs)
                    loss = focal_loss_with_mask(
                        logits=logits,
                        targets=y,
                        masks=m,
                        class_weights=class_weights,
                        alpha=config.alpha_focal,
                        gamma=config.gamma_focal,
                    )
        
                    val_loss_sum += float(loss.item())
                    val_batches += 1
                    val_logits.append(logits)
                    val_targets.append(y)
                    val_masks.append(m)
        
            val_loss = val_loss_sum / max(val_batches, 1)
            history["val_loss"].append(val_loss)
        
            all_val_logits = torch.cat(val_logits, dim=0)
            all_val_targets = torch.cat(val_targets, dim=0)
            all_val_masks = torch.cat(val_masks, dim=0)
        
            val_metrics = compute_metrics(
                logits=all_val_logits,
                targets=all_val_targets,
                masks=all_val_masks,
                threshold=0.5,
            )
        else:
            history["val_loss"].append(None)


        subsection(f"EPOCH {epoch}/{config.num_epochs}")
        if val_metrics is not None:
            print(
                f"[EPOCH {epoch:03d}/{config.num_epochs:03d}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_f1={val_metrics['f1']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )
        else:
            print(
                f"[EPOCH {epoch:03d}/{config.num_epochs:03d}] "
                f"train_loss={train_loss:.4f} "
                f"(validation disabled)"
            )


        subsection("VAL PER-LABEL METRICS")
        table_header("Label", "Prec", "Rec", "F1", "Support")

        for j, stats in enumerate(val_metrics["per_label"]):
            name = _label_name(labels_any[j])
            p = stats["precision"]
            r = stats["recall"]
            f = stats["f1"]
            sup = stats["support"]
            if p is None:
                print(f"{name:20s} {'N/A':>8} {'N/A':>8} {'N/A':>8} {sup:8d}")
            else:
                print(f"{name:20s} {p:8.4f} {r:8.4f} {f:8.4f} {sup:8d}")
        print("-" * 60)

# -------------------------
# Test
# -------------------------
    history["test_loss"] = None
    test_metrics = None
    
    if test_loader is not None:
        model.eval()
        test_loss_sum = 0.0
        test_batches = 0
        test_logits: List[torch.Tensor] = []
        test_targets: List[torch.Tensor] = []
        test_masks: List[torch.Tensor] = []
    
        with torch.no_grad():
            for imgs, y, m in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)
    
                logits = model(imgs)
                loss = focal_loss_with_mask(
                    logits=logits,
                    targets=y,
                    masks=m,
                    class_weights=class_weights,
                    alpha=config.alpha_focal,
                    gamma=config.gamma_focal,
                )
    
                test_loss_sum += float(loss.item())
                test_batches += 1
                test_logits.append(logits)
                test_targets.append(y)
                test_masks.append(m)
    
        test_loss = test_loss_sum / max(test_batches, 1)
        history["test_loss"] = test_loss
        all_test_logits = torch.cat(test_logits, dim=0)
        all_test_targets = torch.cat(test_targets, dim=0)
        all_test_masks = torch.cat(test_masks, dim=0)

        test_metrics: MetricsResult = compute_metrics(
        logits=all_test_logits, targets=all_test_targets, masks=all_test_masks, threshold=0.5
    )
    else:
        print("[TEST] Skipped (no test split)")
    

    #test_loss = test_loss_sum / max(test_batches, 1)
    #history["test_loss"] = test_loss
    print(f"[TEST] test_loss={test_loss:.4f}")



    

    subsection("TEST METRICS (FINAL)")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  F1-score  : {test_metrics['f1']:.4f}")
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(
        f"  TP / FP / TN / FN : "
        f"{test_metrics['tp']} / {test_metrics['fp']} / "
        f"{test_metrics['tn']} / {test_metrics['fn']}"
    )


    subsection("TEST PER-LABEL METRICS")
    table_header("Label", "Prec", "Rec", "F1", "Support")

    for j, stats in enumerate(test_metrics["per_label"]):
        name = _label_name(labels_any[j])
        p = stats["precision"]
        r = stats["recall"]
        f = stats["f1"]
        sup = stats["support"]
        if p is None:
            print(f"{name:20s} {'N/A':>8} {'N/A':>8} {'N/A':>8} {sup:8d}")
        else:
            print(f"{name:20s} {p:8.4f} {r:8.4f} {f:8.4f} {sup:8d}")
    print("-" * 60)

    # Save
    runs_dir = Path(config.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"dataset_{config.dataset_uuid}_{config.backbone_name}_v{config.labelset_version_to_train}_multilabel"
    model_path = runs_dir / f"{run_name}.pth"
    meta_path = runs_dir / f"{run_name}_meta.json"

    torch.save(model.state_dict(), model_path)

    meta: Dict[str, Any] = {
        "config": config.to_ddict(),
        "labelset": labelset_any,
        "used_label_names": [_label_name(lbl) for lbl in labels_any],
        "used_label_indices_original": kept_indices,
        "history": history,
        "test_metrics_final": test_metrics,
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"model_path": str(model_path), "meta_path": str(meta_path), "history": history}
