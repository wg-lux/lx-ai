from __future__ import annotations
from pydantic import field_validator, model_validator
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _read_secret_file(path: Path, label: str) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"Unable to read {label} from {path}: {exc}") from exc

    if not value:
        raise ValueError(f"Secret file for {label} is empty: {path}")

    return value


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DJANGO_",
        case_sensitive=False,
        extra="ignore",
    )

    debug: bool = False

    db_engine: str = "django.db.backends.sqlite3"
    db_name: str = "endoregDbLocal"
    db_user: str = "endoregDbLocal"
    db_password: str = ""
    db_password_file: Path | None = None
    db_host: str = "localhost"
    db_port: str = "5432"
    db_sslmode: str = "disable"

    data_dir: Path = Path("data")
    conf_dir: Path = Path("conf")
    frame_dir: Path = Path("data/frames")
    log_level: str = "INFO"

    @field_validator("db_password_file", mode="before")
    @classmethod
    def normalize_secret_file(cls, value):
        if value is None:
            return None

        if isinstance(value, str) and not value.strip():
            return None

        return value

    @model_validator(mode="after")
    def validate_runtime_configuration(self) -> AppConfig:
        environment = (
            os.getenv(
                "DJANGO_ENV",
                "development",
            )
            .strip()
            .lower()
        )

        is_production = environment in {
            "production",
            "prod",
        }

        is_postgres = self.db_engine.endswith("postgresql")

        # Resolve password file if configured.
        if self.db_password_file and not self.db_password:
            self.db_password = _read_secret_file(
                Path(self.db_password_file),
                "db_password",
            )

        # Production must use PostgreSQL.
        if is_production and not is_postgres:
            raise ValueError(
                f"Production must use PostgreSQL. Configured engine: {self.db_engine}"
            )

        # Never allow production PostgreSQL without credentials.
        if is_production and is_postgres and not self.db_password:
            raise ValueError(
                "Production PostgreSQL password is missing. "
                "Set DJANGO_DB_PASSWORD or provide a valid "
                "DJANGO_DB_PASSWORD_FILE."
            )

        return self


def load_config(
    env_file: Path | None = None,
) -> AppConfig:
    if env_file is not None:
        return AppConfig(_env_file=env_file)

    return AppConfig()


db_backend: str = (
    os.getenv(
        "DB_BACKEND",
        "sqlite",
    )
    .strip()
    .lower()
)
