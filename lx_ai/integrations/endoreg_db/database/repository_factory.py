"""Factory for selecting the EndoRegDB repository."""

import os

from django.db import connection

from lx_ai.integrations.endoreg_db.contracts import DatabaseRepository
from lx_ai.integrations.endoreg_db.database.django_repository import (
    DjangoORMRepository,
)
from lx_ai.integrations.endoreg_db.exceptions import (
    EndoRegDBCompatibilityError,
)


def get_database_repository() -> DatabaseRepository:
    """Return the ORM repository using Django's active connection."""

    actual_vendor = connection.vendor

    if actual_vendor not in {"postgresql", "sqlite"}:
        raise EndoRegDBCompatibilityError(
            f"Unsupported Django database vendor: {actual_vendor!r}"
        )

    configured_backend = os.getenv("DB_BACKEND", "").strip().lower()

    expected_vendors = {
        "postgres": "postgresql",
        "postgresql": "postgresql",
        "sqlite": "sqlite",
    }

    if configured_backend:
        expected_vendor = expected_vendors.get(configured_backend)

        if expected_vendor is None:
            raise EndoRegDBCompatibilityError(
                f"Unsupported DB_BACKEND: {configured_backend!r}"
            )

        if expected_vendor != actual_vendor:
            raise EndoRegDBCompatibilityError(
                "DB_BACKEND does not match Django's active connection: "
                f"configured={configured_backend!r}, "
                f"actual={actual_vendor!r}"
            )

    return DjangoORMRepository()
