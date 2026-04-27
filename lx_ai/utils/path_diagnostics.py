from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from lx_ai.utils.logging_utils import kv, section, subsection, table_header
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