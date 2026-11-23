"""Backward-compatible LX-AI database loading facade.

All database access is delegated to the shared Django ORM repository.
The Django connection is the single source of truth for the backend.
"""

from django.db import connection

from lx_ai.integrations.endoreg_db.database.repository_factory import (
    get_database_repository,
)
from lx_ai.integrations.endoreg_db.exceptions import (
    EndoRegDBCompatibilityError,
)


def _require_vendor(expected: str, operation: str) -> None:
    """Validate legacy backend-specific calls against Django."""

    if connection.vendor != expected:
        raise EndoRegDBCompatibilityError(
            f"{operation} requires {expected}, but Django uses {connection.vendor}."
        )


def load_annotations(config, dataset_id: int):
    """Load annotations using the active Django database."""

    return get_database_repository().get_annotations(dataset_id)


def load_annotations_from_postgres(dataset_id: int):
    """Preserve the existing PostgreSQL loader interface."""

    _require_vendor("postgresql", "load_annotations_from_postgres")

    return get_database_repository().get_annotations(dataset_id)


def load_annotations_from_sqlite(dataset_id: int):
    """Preserve the existing SQLite loader interface."""

    _require_vendor("sqlite", "load_annotations_from_sqlite")

    return get_database_repository().get_annotations(dataset_id)


def load_labelset_from_postgres(
    *,
    labelset_id: int,
    labelset_version: int,
):
    """Preserve the existing PostgreSQL labelset interface."""

    _require_vendor("postgresql", "load_labelset_from_postgres")

    return get_database_repository().get_labelset(
        labelset_id=labelset_id,
        labelset_version=labelset_version,
    )


def load_labelset_from_sqlite(
    labelset_id: int,
    labelset_version: int,
):
    """Preserve the existing SQLite labelset interface."""

    _require_vendor("sqlite", "load_labelset_from_sqlite")

    return get_database_repository().get_labelset(
        labelset_id=labelset_id,
        labelset_version=labelset_version,
    )


__all__ = [
    "load_annotations",
    "load_annotations_from_postgres",
    "load_annotations_from_sqlite",
    "load_labelset_from_postgres",
    "load_labelset_from_sqlite",
]
