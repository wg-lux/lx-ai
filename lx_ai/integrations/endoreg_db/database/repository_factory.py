"""Factory for selecting the EndoRegDB repository."""

import os

from lx_ai.integrations.endoreg_db.contracts import (
    DatabaseRepository,
)
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
    """Return the configured database repository."""

    backend = (
        os.getenv(
            "DB_BACKEND",
            "postgres",
        )
        .strip()
        .lower()
    )

    if backend == "postgres":
        return PostgresRepository()

    if backend == "sqlite":
        return SQLiteRepository()

    raise EndoRegDBCompatibilityError(f"Unsupported database backend: {backend!r}")
