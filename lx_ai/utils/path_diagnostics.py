from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from lx_ai.utils.logging_utils import (
    kv,
    section,
    subsection,
    table_header,
    error,
    success,
    warning,
)
from django.conf import settings
from django.db import connection


PATH_ENV_KEYS: tuple[str, ...] = (
    "HOME_DIR",
    "WORKING_DIR",
    "DATA_DIR",
    "CONF_DIR",
    "STORAGE_DIR",
    "FRAME_DIR",
    "TRAINING_ROOT",
    "CHECKPOINTS_DIR",
    "RUNS_DIR",
    "BUCKET_SNAPSHOT_DIR",
    "BACKBONE_CHECKPOINT",
    "TRAINING_CONFIG_PATH",
    "LEGACY_IMAGE_DIR",
    "LEGACY_JSONL_PATH",
    "CSV_DIR",
    "SQLITE_DB_PATH",
    "FRAME_PATH_REMAP_SOURCE",
    "FRAME_PATH_REMAP_TARGET",
)


DB_ENV_KEYS: tuple[str, ...] = (
    "DB_BACKEND",
    "DJANGO_DB_ENGINE",
    "DJANGO_DB_HOST",
    "DJANGO_DB_PORT",
    "DJANGO_DB_NAME",
    "DJANGO_DB_USER",
    "DJANGO_DB_SSLMODE",
    "DJANGO_DB_PASSWORD_FILE",
    "DEV_DB_HOST",
    "DEV_DB_PORT",
    "DEV_DB_NAME",
    "DEV_DB_USER",
    "DEV_DB_SSLMODE",
    "DEV_DB_PASSWORD_FILE",
)


def _exists_text(value: str | None) -> str:
    if not value:
        return "N/A"

    path = Path(os.path.expandvars(value)).expanduser()

    if path.exists():
        return "yes"

    return "no"


def _resolved_text(value: str | None) -> str:
    if not value:
        return "N/A"

    return str(Path(os.path.expandvars(value)).expanduser())


def _print_env_table(keys: Iterable[str]) -> None:
    table_header("Variable", "Set", "Exists", "Resolved")

    for key in keys:
        value = os.getenv(key)
        is_set = "yes" if value else "no"
        exists = _exists_text(value)
        resolved = _resolved_text(value)

        print(f"{key:<24} {is_set:<10} {exists:<10} {resolved}")


def print_runtime_path_diagnostics() -> None:
    section("RUNTIME PATH DIAGNOSTICS")

    subsection("PATH VARIABLES")
    _print_env_table(PATH_ENV_KEYS)

    subsection("DATABASE VARIABLES")
    _print_env_table(DB_ENV_KEYS)

    subsection("ACTIVE RUNTIME SUMMARY")
    kv("Django settings", os.getenv("DJANGO_SETTINGS_MODULE", "N/A"))
    kv("Django env", os.getenv("DJANGO_ENV", "N/A"))
    kv("DB backend", os.getenv("DB_BACKEND", "N/A"))
    kv("Training config", os.getenv("TRAINING_CONFIG_PATH", "N/A"))
    kv("Data dir", os.getenv("DATA_DIR", "N/A"))
    kv("Frame dir", os.getenv("FRAME_DIR", "N/A"))
    kv("Checkpoint", os.getenv("BACKBONE_CHECKPOINT", "N/A"))

    subsection("ACTIVE DJANGO DATABASE")

    db_cfg = settings.DATABASES["default"]
    kv("Django settings", settings.SETTINGS_MODULE)
    kv("Django DB vendor", connection.vendor)
    kv("Django DB engine", db_cfg.get("ENGINE"))
    kv("Django DB name/path", db_cfg.get("NAME"))


class RuntimePathValidationError(RuntimeError):
    """Raised when required runtime paths are missing."""


def _env_path(name: str, default: str | None = None) -> Path | None:
    value = os.getenv(name, default)
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser()


def _path_exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value.expanduser()
    value_str = str(value).strip()
    if not value_str:
        return None
    return Path(value_str).expanduser()


def validate_runtime_paths_for_training(config: Any) -> None:
    """
    Validate only the paths required for the active training mode.

    This function is intentionally conservative:
    - required active paths fail with a clear RuntimePathValidationError
    - optional paths only print warnings
    - directories needed for outputs are created if missing
    """

    section("RUNTIME PATH VALIDATION")

    missing_required: list[tuple[str, str, str]] = []
    warnings: list[tuple[str, str, str]] = []

    data_dir = _env_path("DATA_DIR")
    training_config_path = _env_path(
        "TRAINING_CONFIG_PATH",
        "lx_ai/ai_model_config/train_sandbox_postgres.yaml",
    )

    training_root = _as_path(_config_value(config, "training_root")) or _env_path(
        "TRAINING_ROOT"
    )
    checkpoints_dir = _as_path(_config_value(config, "checkpoints_dir")) or _env_path(
        "CHECKPOINTS_DIR"
    )
    runs_dir = _as_path(_config_value(config, "runs_dir")) or _env_path("RUNS_DIR")
    bucket_snapshot_dir = _env_path("BUCKET_SNAPSHOT_DIR")

    backbone_name = str(_config_value(config, "backbone_name", ""))
    backbone_checkpoint = _as_path(_config_value(config, "backbone_checkpoint"))
    if backbone_checkpoint is None:
        backbone_checkpoint = _env_path("BACKBONE_CHECKPOINT")

    data_source = str(_config_value(config, "data_source", ""))
    db_backend = os.getenv("DB_BACKEND", "postgres").strip().lower()

    sqlite_db_path = _env_path("SQLITE_DB_PATH")
    password_file = _env_path("DJANGO_DB_PASSWORD_FILE") or _env_path(
        "DEV_DB_PASSWORD_FILE"
    )

    required_existing_paths: list[tuple[str, Path | None, str]] = [
        ("DATA_DIR", data_dir, "main data directory must exist"),
        (
            "TRAINING_CONFIG_PATH",
            training_config_path,
            "training YAML config must exist",
        ),
    ]

    if checkpoints_dir is not None:
        required_existing_paths.append(
            (
                "CHECKPOINTS_DIR",
                checkpoints_dir,
                "checkpoint directory must exist",
            )
        )

    if backbone_name == "gastro_rn50":
        required_existing_paths.append(
            (
                "BACKBONE_CHECKPOINT",
                backbone_checkpoint,
                "required when backbone_name is gastro_rn50",
            )
        )

    if data_source == "jsonl":
        required_existing_paths.extend(
            [
                (
                    "LEGACY_IMAGE_DIR",
                    _env_path("LEGACY_IMAGE_DIR"),
                    "required when data_source is jsonl",
                ),
                (
                    "LEGACY_JSONL_PATH",
                    _env_path("LEGACY_JSONL_PATH"),
                    "required when data_source is jsonl",
                ),
            ]
        )

    if db_backend == "sqlite":
        required_existing_paths.append(
            (
                "SQLITE_DB_PATH",
                sqlite_db_path,
                "required when DB_BACKEND is sqlite",
            )
        )

    if db_backend == "postgres":
        required_existing_paths.append(
            (
                "DJANGO_DB_PASSWORD_FILE",
                password_file,
                "required when DB_BACKEND is postgres",
            )
        )

    for name, path, reason in required_existing_paths:
        if path is None:
            missing_required.append((name, "not set", reason))
        elif not path.exists():
            missing_required.append((name, str(path), reason))

    creatable_dirs: list[tuple[str, Path | None, str]] = [
        ("TRAINING_ROOT", training_root, "training artifact root"),
        ("RUNS_DIR", runs_dir, "model output directory"),
        ("BUCKET_SNAPSHOT_DIR", bucket_snapshot_dir, "bucket snapshot directory"),
    ]

    for name, path, reason in creatable_dirs:
        if path is None:
            missing_required.append((name, "not set", reason))
            continue

        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            missing_required.append(
                (name, str(path), f"{reason}; cannot create: {exc}")
            )

    optional_paths: list[tuple[str, Path | None, str]] = [
        ("CSV_DIR", _env_path("CSV_DIR"), "only required for CSV import"),
        (
            "FRAME_PATH_REMAP_SOURCE",
            _env_path("FRAME_PATH_REMAP_SOURCE"),
            "only required when remapping service frame paths",
        ),
        (
            "FRAME_PATH_REMAP_TARGET",
            _env_path("FRAME_PATH_REMAP_TARGET"),
            "only required when remapping service frame paths",
        ),
    ]

    for name, path, reason in optional_paths:
        if path is None:
            warnings.append((name, "not set", reason))
        elif not path.exists():
            warnings.append((name, str(path), reason))

    subsection("Required path check")
    if missing_required:
        error("Missing required runtime paths:")
        for name, path, reason in missing_required:
            kv(name, path)
            print(f"  Reason: {reason}")

        print()
        error("Fix:")
        print("  1. update .env or service environment")
        print("  2. create missing directories")
        print("  3. place required files such as backbone checkpoint")
        print("  4. rerun path diagnostics")

        raise RuntimePathValidationError(
            "Runtime path validation failed. See missing required paths above."
        )

    success("All required runtime paths are valid.")

    subsection("Optional path check")
    if warnings:
        warning("Some optional paths are missing:")
        for name, path, reason in warnings:
            kv(name, path)
            print(f"  Meaning: {reason}")
    else:
        success("All optional paths are present.")
