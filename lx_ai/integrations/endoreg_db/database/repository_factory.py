"""Factory for selecting the EndoRegDB repository."""

import os

from django.db import connection

from lx_ai.integrations.endoreg_db.contracts import DatabaseRepository
from lx_ai.integrations.endoreg_db.database.postgres_repository import (
    PostgresRepository,
)
from lx_ai.integrations.endoreg_db.database.sqlite_repository import (
    SQLiteRepository,
)
from lx_ai.integrations.endoreg_db.exceptions import (
    EndoRegDBCompatibilityError,
)


def get_database_repository() -> DatabaseRepository:
    """Return the repository matching the active Django database."""

    backend = os.getenv("DB_BACKEND", "").strip().lower()

    expected_vendors = {
        "postgres": "postgresql",
        "sqlite": "sqlite",
    }

    expected_vendor = expected_vendors.get(backend)

    if expected_vendor is None:
        raise EndoRegDBCompatibilityError(f"Unsupported database backend: {backend!r}")

    actual_vendor = connection.vendor

    if actual_vendor != expected_vendor:
        raise EndoRegDBCompatibilityError(
            "Database configuration mismatch: "
            f"DB_BACKEND={backend!r}, "
            f"Django connection.vendor={actual_vendor!r}. "
            "Repository and Django ORM must use the same database."
        )

    if backend == "postgres":
        return PostgresRepository()

    return SQLiteRepository()
