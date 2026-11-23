from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lx_ai.utils import db_loader_for_model_input as db_loader
from lx_ai.integrations.endoreg_db.exceptions import (
    EndoRegDBCompatibilityError,
)


@pytest.fixture
def repository(monkeypatch):
    """Provide a mocked ORM repository."""

    mock_repository = Mock()

    monkeypatch.setattr(
        db_loader,
        "get_database_repository",
        lambda: mock_repository,
    )

    return mock_repository


@pytest.fixture
def postgres_connection(monkeypatch):
    """Simulate an active PostgreSQL Django connection."""

    monkeypatch.setattr(
        db_loader,
        "connection",
        SimpleNamespace(vendor="postgresql"),
    )


@pytest.fixture
def sqlite_connection(monkeypatch):
    """Simulate an active SQLite Django connection."""

    monkeypatch.setattr(
        db_loader,
        "connection",
        SimpleNamespace(vendor="sqlite"),
    )


class TestDatabaseLoader:
    def test_load_annotations_delegates_to_repository(
        self,
        repository,
    ):
        expected = [
            {"annotation_id": 100},
            {"annotation_id": 101},
        ]

        repository.get_annotations.return_value = expected

        result = db_loader.load_annotations(
            config=None,
            dataset_id=10,
        )

        assert result == expected

        repository.get_annotations.assert_called_once_with(10)

    def test_load_annotations_from_postgres(
        self,
        repository,
        postgres_connection,
    ):
        expected = [{"annotation_id": 100}]

        repository.get_annotations.return_value = expected

        result = db_loader.load_annotations_from_postgres(
            dataset_id=10,
        )

        assert result == expected

        repository.get_annotations.assert_called_once_with(10)

    def test_load_annotations_from_sqlite(
        self,
        repository,
        sqlite_connection,
    ):
        expected = [{"annotation_id": 100}]

        repository.get_annotations.return_value = expected

        result = db_loader.load_annotations_from_sqlite(
            dataset_id=10,
        )

        assert result == expected

        repository.get_annotations.assert_called_once_with(10)

    def test_load_labelset_from_postgres(
        self,
        repository,
        postgres_connection,
    ):
        expected = {
            "id": 5,
            "version": 1,
            "labels": [
                {"id": 1, "name": "polyp"},
                {"id": 2, "name": "normal"},
            ],
        }

        repository.get_labelset.return_value = expected

        result = db_loader.load_labelset_from_postgres(
            labelset_id=5,
            labelset_version=1,
        )

        assert result == expected

        repository.get_labelset.assert_called_once_with(
            labelset_id=5,
            labelset_version=1,
        )

    def test_load_labelset_from_sqlite(
        self,
        repository,
        sqlite_connection,
    ):
        expected = {
            "id": 5,
            "version": 1,
            "labels": [
                {"id": 1, "name": "polyp"},
            ],
        }

        repository.get_labelset.return_value = expected

        result = db_loader.load_labelset_from_sqlite(
            labelset_id=5,
            labelset_version=1,
        )

        assert result == expected

        repository.get_labelset.assert_called_once_with(
            labelset_id=5,
            labelset_version=1,
        )


class TestDatabaseBackendCompatibility:
    def test_postgres_loader_rejects_sqlite_connection(
        self,
        repository,
        sqlite_connection,
    ):
        with pytest.raises(
            EndoRegDBCompatibilityError,
            match="requires postgresql",
        ):
            db_loader.load_annotations_from_postgres(
                dataset_id=10,
            )

        repository.get_annotations.assert_not_called()

    def test_sqlite_loader_rejects_postgres_connection(
        self,
        repository,
        postgres_connection,
    ):
        with pytest.raises(
            EndoRegDBCompatibilityError,
            match="requires sqlite",
        ):
            db_loader.load_annotations_from_sqlite(
                dataset_id=10,
            )

        repository.get_annotations.assert_not_called()

    def test_postgres_labelset_rejects_sqlite_connection(
        self,
        repository,
        sqlite_connection,
    ):
        with pytest.raises(
            EndoRegDBCompatibilityError,
            match="requires postgresql",
        ):
            db_loader.load_labelset_from_postgres(
                labelset_id=5,
                labelset_version=1,
            )

        repository.get_labelset.assert_not_called()

    def test_sqlite_labelset_rejects_postgres_connection(
        self,
        repository,
        postgres_connection,
    ):
        with pytest.raises(
            EndoRegDBCompatibilityError,
            match="requires sqlite",
        ):
            db_loader.load_labelset_from_sqlite(
                labelset_id=5,
                labelset_version=1,
            )

        repository.get_labelset.assert_not_called()


def test_public_exports():
    """Verify that existing public loader names remain available."""

    expected_exports = {
        "load_annotations",
        "load_annotations_from_postgres",
        "load_annotations_from_sqlite",
        "load_labelset_from_postgres",
        "load_labelset_from_sqlite",
    }

    assert set(db_loader.__all__) == expected_exports

    for name in expected_exports:
        assert callable(getattr(db_loader, name))
