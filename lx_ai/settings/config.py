from __future__ import annotations

from pathlib import Path
from typing import Annotated
import os

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _read_secret_file(path: Path, label: str) -> str:
    try:
        value = path.read_text().strip()
    except OSError as exc:
        raise ValueError(f"Unable to read {label} from {path}: {exc}") from exc
    if not value:
        raise ValueError(f"Secret file {path} for {label} is empty")
    return value


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DJANGO_",
        case_sensitive=False,
        extra="ignore",
    )

    debug: bool = False

    db_engine: str = "django.db.backends.postgresql"
    db_name: str = os.getenv("DJANGO_DB_NAME", "endoregDbLocal")
    db_user: str = os.getenv("DJANGO_DB_USER", "endoregDbLocal")
    db_password: str = ""
    db_password_file: Path | None = None
    db_host: str = os.getenv("DJANGO_DB_HOST", "localhost")
    db_port: str = os.getenv("DJANGO_DB_PORT", "5432")
    db_sslmode: str = "prefer"

    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    conf_dir: Path = Path(os.getenv("CONF_DIR", "conf"))
    frame_dir: Path = Path(os.getenv("FRAME_DIR", "data/frames"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @field_validator("db_password_file", mode="before")
    @classmethod
    def normalize_secret_file(cls, value):
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def apply_secret_files(self) -> "AppConfig":
        if self.db_password_file and not self.db_password:
            path = Path(self.db_password_file)
    
            # Same behavior as lx-annotate
            if path.exists():
                self.db_password = _read_secret_file(path, "db_password")
    
        return self


def load_config(env_file: Path | None = None) -> AppConfig:
    return AppConfig()

@model_validator(mode="after")
def debug_print(self) -> "AppConfig":
    print("\n========== APP CONFIG DEBUG ==========")
    print(f"DATA_DIR: {self.data_dir}")
    print(f"CONF_DIR: {self.conf_dir}")
    print(f"FRAME_DIR: {self.frame_dir}")
    print(f"DJANGO_ENV: {os.getenv('DJANGO_ENV')}")
    print(f"ENV DATA_DIR: {os.getenv('DATA_DIR')}")
    print("======================================\n")
    print(f"DB PASSWORD FILE: {self.db_password_file}")
    print(f"DB PASSWORD LOADED: {bool(self.db_password)}")
    return self

db_backend: str = os.getenv("DB_BACKEND", "postgres")
