from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lx_ai.utils.path_diagnostics import (
    RuntimePathValidationError,
    validate_runtime_paths_for_training,
)


def _make_config(
    tmp_path: Path,
    *,
    data_source: str = "postgres",
    backbone_name: str = "gastro_rn50",
    backbone_checkpoint: Path | None = None,
) -> SimpleNamespace:
    # creates minimal config object needed by runtime path validation
    training_root = tmp_path / "data" / "model_training"
    checkpoints_dir = training_root / "checkpoints"
    runs_dir = training_root / "runs"

    return SimpleNamespace(
        data_source=data_source,
        backbone_name=backbone_name,
        backbone_checkpoint=backbone_checkpoint,
        training_root=training_root,
        checkpoints_dir=checkpoints_dir,
        runs_dir=runs_dir,
    )


def _prepare_common_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    # creates the common required local directories and files
    data_dir = tmp_path / "data"
    conf_dir = tmp_path / "conf"
    training_root = data_dir / "model_training"
    checkpoints_dir = training_root / "checkpoints"
    runs_dir = training_root / "runs"
    bucket_dir = training_root / "buckets"
    config_path = tmp_path / "train.yaml"
    checkpoint_path = checkpoints_dir / "RN50_GastroNet-1M_DINOv1.pth"
    sqlite_db = tmp_path / "dev_db.sqlite"
    password_file = conf_dir / "db_pwd"

    data_dir.mkdir(parents=True)
    conf_dir.mkdir(parents=True)
    checkpoints_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)
    bucket_dir.mkdir(parents=True)

    config_path.write_text("dataset_uuid: test\n", encoding="utf-8")
    checkpoint_path.write_bytes(b"fake-checkpoint")
    sqlite_db.write_text("", encoding="utf-8")
    password_file.write_text("secret", encoding="utf-8")

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("CONF_DIR", str(conf_dir))
    monkeypatch.setenv("TRAINING_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("TRAINING_ROOT", str(training_root))
    monkeypatch.setenv("CHECKPOINTS_DIR", str(checkpoints_dir))
    monkeypatch.setenv("RUNS_DIR", str(runs_dir))
    monkeypatch.setenv("BUCKET_SNAPSHOT_DIR", str(bucket_dir))
    monkeypatch.setenv("BACKBONE_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setenv("SQLITE_DB_PATH", str(sqlite_db))
    monkeypatch.setenv("DJANGO_DB_PASSWORD_FILE", str(password_file))

    return {
        "data_dir": data_dir,
        "conf_dir": conf_dir,
        "training_root": training_root,
        "checkpoints_dir": checkpoints_dir,
        "runs_dir": runs_dir,
        "bucket_dir": bucket_dir,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "sqlite_db": sqlite_db,
        "password_file": password_file,
    }


class TestRuntimePathValidation:
    def test_validation_passes_for_sqlite_when_required_paths_exist(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # checks sqlite mode passes when config, checkpoint and sqlite db exist
        paths = _prepare_common_paths(tmp_path, monkeypatch)
        monkeypatch.setenv("DB_BACKEND", "sqlite")

        config = _make_config(
            tmp_path,
            backbone_checkpoint=paths["checkpoint_path"],
        )

        validate_runtime_paths_for_training(config)

    def test_validation_passes_for_postgres_when_password_file_exists(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # checks postgres mode passes when password file exists
        paths = _prepare_common_paths(tmp_path, monkeypatch)
        monkeypatch.setenv("DB_BACKEND", "postgres")

        config = _make_config(
            tmp_path,
            backbone_checkpoint=paths["checkpoint_path"],
        )

        validate_runtime_paths_for_training(config)

    def test_validation_fails_when_training_config_is_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # checks missing training yaml fails early with clear error
        paths = _prepare_common_paths(tmp_path, monkeypatch)
        monkeypatch.setenv("DB_BACKEND", "sqlite")
        monkeypatch.setenv("TRAINING_CONFIG_PATH", str(tmp_path / "missing.yaml"))

        config = _make_config(
            tmp_path,
            backbone_checkpoint=paths["checkpoint_path"],
        )

        with pytest.raises(RuntimePathValidationError):
            validate_runtime_paths_for_training(config)

    def test_validation_fails_when_backbone_checkpoint_is_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # checks gastro_rn50 requires the checkpoint file
        _prepare_common_paths(tmp_path, monkeypatch)
        monkeypatch.setenv("DB_BACKEND", "sqlite")

        missing_checkpoint = tmp_path / "missing_checkpoint.pth"

        config = _make_config(
            tmp_path,
            backbone_name="gastro_rn50",
            backbone_checkpoint=missing_checkpoint,
        )

        with pytest.raises(RuntimePathValidationError):
            validate_runtime_paths_for_training(config)

    def test_validation_allows_missing_checkpoint_for_non_gastro_backbone(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # checks non gastro backbone does not require BACKBONE_CHECKPOINT
        _prepare_common_paths(tmp_path, monkeypatch)
        monkeypatch.setenv("DB_BACKEND", "sqlite")
        monkeypatch.delenv("BACKBONE_CHECKPOINT", raising=False)

        config = _make_config(
            tmp_path,
            backbone_name="resnet50_imagenet",
            backbone_checkpoint=None,
        )

        validate_runtime_paths_for_training(config)

    def test_validation_fails_when_sqlite_db_is_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # checks sqlite mode requires SQLITE_DB_PATH
        paths = _prepare_common_paths(tmp_path, monkeypatch)
        monkeypatch.setenv("DB_BACKEND", "sqlite")
        monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "missing.sqlite"))

        config = _make_config(
            tmp_path,
            backbone_checkpoint=paths["checkpoint_path"],
        )

        with pytest.raises(RuntimePathValidationError):
            validate_runtime_paths_for_training(config)

    def test_validation_fails_when_postgres_password_file_is_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # checks postgres mode requires password file
        paths = _prepare_common_paths(tmp_path, monkeypatch)
        monkeypatch.setenv("DB_BACKEND", "postgres")
        monkeypatch.setenv("DJANGO_DB_PASSWORD_FILE", str(tmp_path / "missing_pwd"))

        config = _make_config(
            tmp_path,
            backbone_checkpoint=paths["checkpoint_path"],
        )

        with pytest.raises(RuntimePathValidationError):
            validate_runtime_paths_for_training(config)

    def test_validation_creates_output_directories_when_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # checks output directories are created instead of failing
        paths = _prepare_common_paths(tmp_path, monkeypatch)
        monkeypatch.setenv("DB_BACKEND", "sqlite")

        runs_dir = tmp_path / "new_runs"
        bucket_dir = tmp_path / "new_buckets"

        monkeypatch.setenv("RUNS_DIR", str(runs_dir))
        monkeypatch.setenv("BUCKET_SNAPSHOT_DIR", str(bucket_dir))

        config = SimpleNamespace(
            data_source="postgres",
            backbone_name="gastro_rn50",
            backbone_checkpoint=paths["checkpoint_path"],
            training_root=tmp_path / "new_training_root",
            checkpoints_dir=paths["checkpoints_dir"],
            runs_dir=runs_dir,
        )

        validate_runtime_paths_for_training(config)

        assert runs_dir.exists()
        assert bucket_dir.exists()

    def test_optional_paths_missing_do_not_fail(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # checks optional CSV and remap paths only warn and do not fail
        paths = _prepare_common_paths(tmp_path, monkeypatch)
        monkeypatch.setenv("DB_BACKEND", "sqlite")

        monkeypatch.setenv("CSV_DIR", str(tmp_path / "missing_csv"))
        monkeypatch.setenv("FRAME_PATH_REMAP_SOURCE", str(tmp_path / "missing_source"))
        monkeypatch.setenv("FRAME_PATH_REMAP_TARGET", str(tmp_path / "missing_target"))

        config = _make_config(
            tmp_path,
            backbone_checkpoint=paths["checkpoint_path"],
        )

        validate_runtime_paths_for_training(config)

    def test_jsonl_mode_requires_legacy_paths(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # checks jsonl mode requires image directory and jsonl file
        paths = _prepare_common_paths(tmp_path, monkeypatch)
        monkeypatch.setenv("DB_BACKEND", "sqlite")
        monkeypatch.setenv("LEGACY_IMAGE_DIR", str(tmp_path / "missing_images"))
        monkeypatch.setenv("LEGACY_JSONL_PATH", str(tmp_path / "missing.jsonl"))

        config = _make_config(
            tmp_path,
            data_source="jsonl",
            backbone_checkpoint=paths["checkpoint_path"],
        )

        with pytest.raises(RuntimePathValidationError):
            validate_runtime_paths_for_training(config)

    def test_jsonl_mode_passes_when_legacy_paths_exist(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # checks jsonl mode passes when image directory and jsonl file exist
        paths = _prepare_common_paths(tmp_path, monkeypatch)
        monkeypatch.setenv("DB_BACKEND", "sqlite")

        image_dir = tmp_path / "legacy_images" / "images"
        jsonl_path = tmp_path / "legacy_images" / "legacy_img_dicts.jsonl"

        image_dir.mkdir(parents=True)
        jsonl_path.write_text(
            '{"labels": ["polyp"], "old_id": 1, "filename": "1.jpg"}\n',
            encoding="utf-8",
        )

        monkeypatch.setenv("LEGACY_IMAGE_DIR", str(image_dir))
        monkeypatch.setenv("LEGACY_JSONL_PATH", str(jsonl_path))

        config = _make_config(
            tmp_path,
            data_source="jsonl",
            backbone_checkpoint=paths["checkpoint_path"],
        )

        validate_runtime_paths_for_training(config)