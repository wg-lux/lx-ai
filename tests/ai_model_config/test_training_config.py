from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lx_ai.ai_model_config.config import TrainingConfig


def _base_config_dict(tmp_path: Path) -> dict:
    # Base valid config used in all tests.
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fake-checkpoint")

    return {
        "dataset_uuid": "sandbox_ds",
        "data_source": "postgres",
        "dataset_ids": [1, 2],
        "labelset_id": 5,
        "labelset_version_to_train": 3,
        "treat_unlabeled_as_negative": False,
        "base_dir": str(tmp_path),
        "training_root": str(tmp_path / "training"),
        "checkpoints_dir": str(tmp_path / "training" / "checkpoints"),
        "runs_dir": str(tmp_path / "training" / "runs"),
        "create_dirs": True,
        "backbone_name": "gastro_rn50",
        "backbone_checkpoint": str(checkpoint),
        "freeze_backbone": True,
        "num_epochs": 2,
        "batch_size": 3,
        "lr_head": 0.001,
        "lr_backbone": 0.0001,
        "gamma_focal": 2.0,
        "alpha_focal": 0.25,
        "use_scheduler": True,
        "warmup_epochs": 2,
        "min_lr": 1.0e-6,
        "device": "cpu",
        "random_seed": 42,
        "bucket_policy": {
            "num_buckets": 5,
            "validation_buckets": [3],
            "test_buckets": [4],
        },
        "save_bucket_snapshot": True,
    }


# -----------------------------------------------------------------------------
# PostgreSQL source
# -----------------------------------------------------------------------------


def test_valid_postgres_config_creates_dirs(
    tmp_path: Path,
) -> None:
    # Checks that valid postgres config works and directories are created.
    cfg = TrainingConfig.model_validate(
        _base_config_dict(tmp_path)
    )

    assert cfg.data_source == "postgres"
    assert cfg.dataset_ids == [1, 2]
    assert cfg.labelset_id == 5
    assert cfg.base_dir == tmp_path.resolve()
    assert cfg.training_root.exists()
    assert cfg.checkpoints_dir.exists()
    assert cfg.runs_dir.exists()


def test_postgres_requires_dataset_ids(
    tmp_path: Path,
) -> None:
    # Checks that dataset_ids is required in postgres mode.
    data = _base_config_dict(tmp_path)
    data["dataset_ids"] = None

    with pytest.raises(
        ValidationError,
        match="dataset_ids must be provided",
    ):
        TrainingConfig.model_validate(data)


def test_postgres_requires_labelset_id(
    tmp_path: Path,
) -> None:
    # Checks that labelset_id is required in postgres mode.
    data = _base_config_dict(tmp_path)
    data["labelset_id"] = None

    with pytest.raises(
        ValidationError,
        match="labelset_id must be provided",
    ):
        TrainingConfig.model_validate(data)


def test_postgres_config_remains_independent_of_manifest_input(
    tmp_path: Path,
) -> None:
    # Runtime manifest input must not affect existing postgres configuration.
    cfg = TrainingConfig.model_validate(
        _base_config_dict(tmp_path)
    )

    assert cfg.data_source == "postgres"
    assert cfg.dataset_ids == [1, 2]
    assert cfg.labelset_id == 5

# -----------------------------------------------------------------------------
# JSONL source
# -----------------------------------------------------------------------------


def test_jsonl_requires_jsonl_path(
    tmp_path: Path,
) -> None:
    # Checks that jsonl mode requires jsonl_path.
    data = _base_config_dict(tmp_path)
    data["data_source"] = "jsonl"
    data["dataset_ids"] = None
    data["labelset_id"] = None
    data["jsonl_path"] = None

    with pytest.raises(
        ValidationError,
        match="jsonl_path must be set",
    ):
        TrainingConfig.model_validate(data)


def test_valid_jsonl_config(
    tmp_path: Path,
) -> None:
    # Checks that valid jsonl config passes.
    jsonl_path = tmp_path / "data.jsonl"

    jsonl_path.write_text(
        (
            '{"labels": ["polyp"], '
            '"old_examination_id": 1, '
            '"old_id": 10, '
            '"filename": "10.jpg"}\n'
        ),
        encoding="utf-8",
    )

    data = _base_config_dict(tmp_path)

    data["data_source"] = "jsonl"
    data["dataset_ids"] = None
    data["labelset_id"] = None
    data["jsonl_path"] = str(jsonl_path)

    cfg = TrainingConfig.model_validate(data)

    assert cfg.data_source == "jsonl"
    assert cfg.jsonl_path == jsonl_path


def test_jsonl_config_remains_independent_of_manifest_input(
    tmp_path: Path,
) -> None:
    jsonl_path = tmp_path / "data.jsonl"

    data = _base_config_dict(tmp_path)

    data["data_source"] = "jsonl"
    data["dataset_ids"] = None
    data["labelset_id"] = None
    data["jsonl_path"] = str(jsonl_path)

    cfg = TrainingConfig.model_validate(data)

    assert cfg.data_source == "jsonl"
    assert cfg.jsonl_path == jsonl_path


# -----------------------------------------------------------------------------
# EndoReg manifest source
# -----------------------------------------------------------------------------
def test_valid_endoreg_manifest_config(
    tmp_path: Path,
) -> None:
    """
    Manifest mode is selected through TrainingConfig, but the actual
    manifest is supplied at runtime to the LX-AI data-loading/API boundary.
    """

    data = _base_config_dict(tmp_path)

    data["data_source"] = "endoreg_manifest"

    # EndoRegDB already resolves dataset and LabelSet domain data before
    # producing the manifest.
    data["dataset_ids"] = None
    data["labelset_id"] = None

    cfg = TrainingConfig.model_validate(data)

    assert cfg.data_source == "endoreg_manifest"
    assert cfg.dataset_ids is None
    assert cfg.labelset_id is None


def test_endoreg_manifest_config_has_no_manifest_path_field(
    tmp_path: Path,
) -> None:
    """
    The manifest is runtime input, not a TrainingConfig filesystem path.
    """

    data = _base_config_dict(tmp_path)

    data["data_source"] = "endoreg_manifest"
    data["dataset_ids"] = None
    data["labelset_id"] = None

    cfg = TrainingConfig.model_validate(data)

    assert not hasattr(
        cfg,
        "endoreg_manifest_path",
    )


# -----------------------------------------------------------------------------
# Checkpoint configuration
# -----------------------------------------------------------------------------


def test_missing_checkpoint_fails(
    tmp_path: Path,
) -> None:
    # Checks that missing checkpoint file raises error.
    data = _base_config_dict(tmp_path)

    data["backbone_checkpoint"] = str(
        tmp_path / "missing.pth"
    )

    with pytest.raises(
        ValidationError,
        match="backbone_checkpoint does not exist",
    ):
        TrainingConfig.model_validate(data)


def test_no_checkpoint_is_allowed(
    tmp_path: Path,
) -> None:
    # Checks that checkpoint can be None.
    data = _base_config_dict(tmp_path)
    data["backbone_checkpoint"] = None

    cfg = TrainingConfig.model_validate(data)

    assert cfg.backbone_checkpoint is None


def test_checkpoint_env_var_expands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Checks that env variable in checkpoint path is expanded correctly.
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()

    checkpoint = (
        checkpoint_dir / "model.pth"
    )

    checkpoint.write_bytes(
        b"fake-checkpoint"
    )

    monkeypatch.setenv(
        "TEST_CKPT_DIR",
        str(checkpoint_dir),
    )

    data = _base_config_dict(tmp_path)

    data["backbone_checkpoint"] = (
        "$TEST_CKPT_DIR/model.pth"
    )

    cfg = TrainingConfig.model_validate(data)

    assert (
        cfg.backbone_checkpoint
        == checkpoint.resolve()
    )


# -----------------------------------------------------------------------------
# General validation
# -----------------------------------------------------------------------------


def test_invalid_backbone_name_fails(
    tmp_path: Path,
) -> None:
    # Checks invalid backbone name is rejected.
    data = _base_config_dict(tmp_path)
    data["backbone_name"] = "invalid_backbone"

    with pytest.raises(
        ValidationError
    ):
        TrainingConfig.model_validate(data)


def test_invalid_device_fails(
    tmp_path: Path,
) -> None:
    # Checks invalid device value is rejected.
    data = _base_config_dict(tmp_path)
    data["device"] = "gpu"

    with pytest.raises(
        ValidationError
    ):
        TrainingConfig.model_validate(data)


def test_invalid_alpha_focal_fails(
    tmp_path: Path,
) -> None:
    # Checks alpha_focal must be in valid range.
    data = _base_config_dict(tmp_path)
    data["alpha_focal"] = 1.5

    with pytest.raises(
        ValidationError
    ):
        TrainingConfig.model_validate(data)


def test_invalid_num_epochs_fails(
    tmp_path: Path,
) -> None:
    # Checks num_epochs must be greater than 0.
    data = _base_config_dict(tmp_path)
    data["num_epochs"] = 0

    with pytest.raises(
        ValidationError
    ):
        TrainingConfig.model_validate(data)


def test_extra_fields_are_forbidden(
    tmp_path: Path,
) -> None:
    # Checks unknown fields are not allowed.
    data = _base_config_dict(tmp_path)

    data["unexpected_field"] = (
        "not allowed"
    )

    with pytest.raises(
        ValidationError
    ):
        TrainingConfig.model_validate(data)


# -----------------------------------------------------------------------------
# Bucket policy
# -----------------------------------------------------------------------------


def test_bucket_policy_train_buckets(
    tmp_path: Path,
) -> None:
    # Checks train buckets are computed correctly from policy.
    cfg = TrainingConfig.model_validate(
        _base_config_dict(tmp_path)
    )

    assert (
        cfg.bucket_policy.train_buckets
        == [0, 1, 2]
    )


def test_overlapping_bucket_policy_fails(
    tmp_path: Path,
) -> None:
    # Checks validation and test buckets must not overlap.
    data = _base_config_dict(tmp_path)

    data["bucket_policy"] = {
        "num_buckets": 5,
        "validation_buckets": [3],
        "test_buckets": [3],
    }

    with pytest.raises(
        ValidationError,
        match="must not overlap",
    ):
        TrainingConfig.model_validate(data)


# -----------------------------------------------------------------------------
# Serialization
# -----------------------------------------------------------------------------


def test_to_ddict_paths_are_strings(
    tmp_path: Path,
) -> None:
    # Checks that to_ddict converts existing paths to strings.
    cfg = TrainingConfig.model_validate(
        _base_config_dict(tmp_path)
    )

    ddict = cfg.to_ddict()

    assert isinstance(
        ddict["base_dir"],
        str,
    )

    assert isinstance(
        ddict["training_root"],
        str,
    )

    assert isinstance(
        ddict["checkpoints_dir"],
        str,
    )

    assert isinstance(
        ddict["runs_dir"],
        str,
    )

    assert isinstance(
        ddict["backbone_checkpoint"],
        str,
    )

    assert isinstance(
        ddict["updated_at"],
        str,
    )


def test_to_ddict_has_no_manifest_transport_path(
    tmp_path: Path,
) -> None:
    data = _base_config_dict(tmp_path)

    data["data_source"] = "endoreg_manifest"
    data["dataset_ids"] = None
    data["labelset_id"] = None

    cfg = TrainingConfig.model_validate(
        data
    )

    ddict = cfg.to_ddict()

    assert (
        "endoreg_manifest_path"
        not in ddict
    )
# -----------------------------------------------------------------------------
# YAML loading
# -----------------------------------------------------------------------------


def test_from_yaml_file(
    tmp_path: Path,
) -> None:
    # Checks that existing postgres config can still be loaded from YAML.
    checkpoint = (
        tmp_path / "checkpoint.pth"
    )

    checkpoint.write_bytes(
        b"fake-checkpoint"
    )

    yaml_path = (
        tmp_path / "train_config.yaml"
    )

    yaml_path.write_text(
        f"""
dataset_uuid: sandbox_ds
data_source: postgres
dataset_ids: [1, 2]
labelset_id: 5
labelset_version_to_train: 3
treat_unlabeled_as_negative: false

base_dir: "{tmp_path}"
training_root: "{tmp_path / "training"}"
checkpoints_dir: "{tmp_path / "training" / "checkpoints"}"
runs_dir: "{tmp_path / "training" / "runs"}"
create_dirs: true

backbone_name: gastro_rn50
backbone_checkpoint: "{checkpoint}"
freeze_backbone: true

num_epochs: 2
batch_size: 3

lr_head: 0.001
lr_backbone: 0.0001
gamma_focal: 2.0
alpha_focal: 0.25

use_scheduler: true
warmup_epochs: 2
min_lr: 1.0e-6

device: cpu
random_seed: 42

bucket_policy:
  num_buckets: 5
  validation_buckets: [3]
  test_buckets: [4]

save_bucket_snapshot: true
""",
        encoding="utf-8",
    )

    cfg = TrainingConfig.from_yaml_file(
        yaml_path
    )

    assert (
        cfg.dataset_uuid
        == "sandbox_ds"
    )

    assert (
        cfg.data_source
        == "postgres"
    )

    assert cfg.dataset_ids == [
        1,
        2,
    ]

    assert cfg.labelset_id == 5

    assert (
        cfg.backbone_checkpoint
        == checkpoint.resolve()
    )


def test_endoreg_manifest_from_yaml_file(
    tmp_path: Path,
) -> None:
    """
    Verify that YAML can select endoreg_manifest mode without embedding
    the manifest or its transport path in TrainingConfig.
    """

    checkpoint = (
        tmp_path / "checkpoint.pth"
    )

    checkpoint.write_bytes(
        b"fake-checkpoint"
    )

    yaml_path = (
        tmp_path
        / "manifest_train_config.yaml"
    )

    yaml_path.write_text(
        f"""
dataset_uuid: sandbox_ds
data_source: endoreg_manifest
labelset_version_to_train: 3
treat_unlabeled_as_negative: false

base_dir: "{tmp_path}"
training_root: "{tmp_path / "training"}"
checkpoints_dir: "{tmp_path / "training" / "checkpoints"}"
runs_dir: "{tmp_path / "training" / "runs"}"
create_dirs: true

backbone_name: gastro_rn50
backbone_checkpoint: "{checkpoint}"
freeze_backbone: true

num_epochs: 2
batch_size: 3

lr_head: 0.001
lr_backbone: 0.0001
gamma_focal: 2.0
alpha_focal: 0.25

use_scheduler: true
warmup_epochs: 2
min_lr: 1.0e-6

device: cpu
random_seed: 42

bucket_policy:
  num_buckets: 5
  validation_buckets: [3]
  test_buckets: [4]

save_bucket_snapshot: true
""",
        encoding="utf-8",
    )

    cfg = TrainingConfig.from_yaml_file(
        yaml_path
    )

    assert (
        cfg.data_source
        == "endoreg_manifest"
    )

    assert cfg.dataset_ids is None
    assert cfg.labelset_id is None

    assert not hasattr(
        cfg,
        "endoreg_manifest_path",
    )