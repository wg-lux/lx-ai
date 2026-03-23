# tests/ai_model_config/test_training_config.py

from pathlib import Path

import pytest

from lx_ai.ai_model_config.config import TrainingConfig


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def minimal_jsonl_config(tmp_path: Path) -> dict:
    """
    Minimal valid configuration for jsonl-based training.

    This represents the smallest config that should pass validation.
    """
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text("{}\n", encoding="utf-8")

    return {
        "dataset_uuid": "test_ds",
        "data_source": "jsonl",
        "jsonl_path": jsonl,
    }


# -----------------------------------------------------------------------------
# 1. Minimal valid config
# -----------------------------------------------------------------------------

def test_training_config__minimal_jsonl_config__passes(tmp_path: Path) -> None:
    """
    GIVEN a minimal jsonl-based configuration
    WHEN TrainingConfig is created
    THEN validation succeeds and defaults are applied
    """
    cfg = TrainingConfig(**minimal_jsonl_config(tmp_path))

    assert cfg.dataset_uuid == "test_ds"
    assert cfg.data_source == "jsonl"
    assert cfg.treat_unlabeled_as_negative is True
    assert cfg.labelset_version_to_train == TrainingConfig.DEFAULT_LABELSET_VERSION


# -----------------------------------------------------------------------------
# 2. Required fields
# -----------------------------------------------------------------------------

def test_training_config__missing_dataset_uuid__raises_error(tmp_path: Path) -> None:
    """
    dataset_uuid is mandatory for all training runs.
    """
    data = minimal_jsonl_config(tmp_path)
    data.pop("dataset_uuid")

    with pytest.raises(Exception):
        TrainingConfig(**data)


# -----------------------------------------------------------------------------
# 3. Data source semantics
# -----------------------------------------------------------------------------

def test_training_config__postgres_without_dataset_id__fails(tmp_path: Path) -> None:
    """
    postgres mode requires dataset_id and labelset_id.
    """
    data = {
        "dataset_uuid": "ds",
        "data_source": "postgres",
        "labelset_id": 1,
    }

    with pytest.raises(ValueError, match="dataset_id must be set"):
        TrainingConfig(**data)


def test_training_config__postgres_without_labelset_id__fails(tmp_path: Path) -> None:
    """
    postgres mode requires an explicit labelset_id.
    """
    data = {
        "dataset_uuid": "ds",
        "data_source": "postgres",
        "dataset_id": 1,
    }

    with pytest.raises(ValueError, match="labelset_id must be provided"):
        TrainingConfig(**data)


# -----------------------------------------------------------------------------
# 4. Path inference and directory creation
# -----------------------------------------------------------------------------

def test_training_config__paths_are_inferred_and_created(tmp_path: Path) -> None:
    """
    If base_dir is provided, training_root, checkpoints_dir and runs_dir
    are derived automatically and created on disk.
    """
    data = minimal_jsonl_config(tmp_path)
    data["base_dir"] = tmp_path

    cfg = TrainingConfig(**data)

    assert cfg.training_root == tmp_path / "data" / "model_training"
    assert cfg.checkpoints_dir.exists()
    assert cfg.runs_dir.exists()


# -----------------------------------------------------------------------------
# 5. Split validation
# -----------------------------------------------------------------------------

def test_training_config__invalid_split_sum__fails(tmp_path: Path) -> None:
    """
    val_split + test_split must leave room for training data.
    """
    data = minimal_jsonl_config(tmp_path)
    data.update({"val_split": 0.6, "test_split": 0.4})

    with pytest.raises(ValueError, match="val_split \\+ test_split"):
        TrainingConfig(**data)


# -----------------------------------------------------------------------------
# 6. Backbone checkpoint validation
# -----------------------------------------------------------------------------

def test_training_config__nonexistent_checkpoint__fails(tmp_path: Path) -> None:
    """
    If a backbone checkpoint is provided, it must exist on disk.
    """
    data = minimal_jsonl_config(tmp_path)
    data["backbone_checkpoint"] = tmp_path / "missing.pth"

    with pytest.raises(ValueError, match="backbone_checkpoint does not exist"):
        TrainingConfig(**data)


def test_training_config__existing_checkpoint__passes(tmp_path: Path) -> None:
    """
    Existing checkpoint paths are accepted and normalized.
    """
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"dummy")

    data = minimal_jsonl_config(tmp_path)
    data["backbone_checkpoint"] = ckpt

    cfg = TrainingConfig(**data)

    assert cfg.backbone_checkpoint == ckpt.resolve()


# -----------------------------------------------------------------------------
# 7. ddict serialization
# -----------------------------------------------------------------------------

def test_training_config__to_ddict__paths_are_strings(tmp_path: Path) -> None:
    """
    to_ddict() produces a JSON-safe representation with paths as strings.
    """
    cfg = TrainingConfig(**minimal_jsonl_config(tmp_path))
    dd = cfg.to_ddict()

    assert isinstance(dd["base_dir"], str)
    assert isinstance(dd["training_root"], str)
    assert isinstance(dd["checkpoints_dir"], str)
    assert isinstance(dd["runs_dir"], str)
