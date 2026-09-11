from __future__ import annotations

from pathlib import Path

import pytest

from lx_ai.utils import db_loader_for_model_input as db_loader


class TestDbLoaderEnvironment:
    def _clean_password_env(self, monkeypatch) -> None:
        # clears all password env vars so tests do not depend on developer or CI shell
        for key in (
            "DEV_DB_PASSWORD",
            "DJANGO_DB_PASSWORD",
            "DEV_DB_PASSWORD_FILE",
            "DJANGO_DB_PASSWORD_FILE",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_first_env_returns_first_existing_value(self, monkeypatch) -> None:
        # checks first non empty env value is returned
        monkeypatch.setenv("FIRST_TEST_ENV", "abc")
        monkeypatch.setenv("SECOND_TEST_ENV", "def")

        value = db_loader._first_env("FIRST_TEST_ENV", "SECOND_TEST_ENV")

        assert value == "abc"

    def test_first_env_skips_empty_values(self, monkeypatch) -> None:
        # checks empty env value is ignored
        monkeypatch.setenv("FIRST_TEST_ENV", "")
        monkeypatch.setenv("SECOND_TEST_ENV", "def")

        value = db_loader._first_env("FIRST_TEST_ENV", "SECOND_TEST_ENV")

        assert value == "def"

    def test_first_env_returns_default_when_missing(self, monkeypatch) -> None:
        # checks default is returned when env variables are missing
        monkeypatch.delenv("MISSING_TEST_ENV", raising=False)

        value = db_loader._first_env("MISSING_TEST_ENV", default="fallback")

        assert value == "fallback"

    def test_get_password_prefers_dev_password(self, monkeypatch) -> None:
        # checks DEV_DB_PASSWORD is preferred
        monkeypatch.setenv("DEV_DB_PASSWORD", "local-secret")
        monkeypatch.setenv("DJANGO_DB_PASSWORD", "service-secret")

        assert db_loader._get_password() == "local-secret"

    def test_get_password_uses_django_password_when_dev_missing(
        self, monkeypatch
    ) -> None:
        # checks DJANGO_DB_PASSWORD is used when DEV_DB_PASSWORD is missing
        monkeypatch.delenv("DEV_DB_PASSWORD", raising=False)
        monkeypatch.setenv("DJANGO_DB_PASSWORD", "service-secret")

        assert db_loader._get_password() == "service-secret"

    def test_get_password_reads_password_file(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        # checks password can be loaded from password file
        self._clean_password_env(monkeypatch)

        password_file = tmp_path / "db_pwd"
        password_file.write_text("file-secret\n", encoding="utf-8")

        monkeypatch.setenv("DJANGO_DB_PASSWORD_FILE", str(password_file))

        assert db_loader._get_password() == "file-secret"

    def test_get_password_raises_when_password_file_missing(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        # checks missing password file raises clear error
        self._clean_password_env(monkeypatch)

        monkeypatch.setenv("DJANGO_DB_PASSWORD_FILE", str(tmp_path / "missing"))

        with pytest.raises(RuntimeError, match="Password file"):
            db_loader._get_password()

    def test_get_password_raises_when_nothing_is_configured(self, monkeypatch) -> None:
        # checks error is raised when no password source exists
        self._clean_password_env(monkeypatch)

        with pytest.raises(RuntimeError, match="No database password found"):
            db_loader._get_password()

    def test_get_db_connection_kwargs_uses_dev_values(self, monkeypatch) -> None:
        # checks connection kwargs are built from DEV_DB values
        monkeypatch.setenv("DEV_DB_HOST", "localhost")
        monkeypatch.setenv("DEV_DB_PORT", "5433")
        monkeypatch.setenv("DEV_DB_NAME", "dev_db")
        monkeypatch.setenv("DEV_DB_USER", "dev_user")
        monkeypatch.setenv("DEV_DB_PASSWORD", "dev_password")
        monkeypatch.setenv("DEV_DB_SSLMODE", "disable")

        kwargs = db_loader._get_db_connection_kwargs()

        assert kwargs == {
            "host": "localhost",
            "port": 5433,
            "dbname": "dev_db",
            "user": "dev_user",
            "password": "dev_password",
            "sslmode": "disable",
        }

    def test_get_db_connection_kwargs_falls_back_to_django_values(
        self,
        monkeypatch,
    ) -> None:
        # checks connection kwargs can use DJANGO_DB values for service mode
        for key in (
            "DEV_DB_HOST",
            "DEV_DB_PORT",
            "DEV_DB_NAME",
            "DEV_DB_USER",
            "DEV_DB_PASSWORD",
            "DEV_DB_SSLMODE",
        ):
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("DJANGO_DB_HOST", "service-host")
        monkeypatch.setenv("DJANGO_DB_PORT", "5432")
        monkeypatch.setenv("DJANGO_DB_NAME", "service_db")
        monkeypatch.setenv("DJANGO_DB_USER", "service_user")
        monkeypatch.setenv("DJANGO_DB_PASSWORD", "service_password")
        monkeypatch.setenv("DJANGO_DB_SSLMODE", "prefer")

        kwargs = db_loader._get_db_connection_kwargs()

        assert kwargs["host"] == "service-host"
        assert kwargs["port"] == 5432
        assert kwargs["dbname"] == "service_db"
        assert kwargs["user"] == "service_user"
        assert kwargs["password"] == "service_password"
        assert kwargs["sslmode"] == "prefer"

    def test_get_db_connection_kwargs_rejects_invalid_port(self, monkeypatch) -> None:
        # checks invalid database port raises clear error
        monkeypatch.setenv("DEV_DB_PORT", "not-an-int")
        monkeypatch.setenv("DEV_DB_PASSWORD", "secret")

        with pytest.raises(RuntimeError, match="Invalid database port"):
            db_loader._get_db_connection_kwargs()
