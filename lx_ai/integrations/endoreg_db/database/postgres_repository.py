"""Backward-compatible PostgreSQL repository.

Database access is implemented centrally by DjangoORMRepository.
The active Django connection determines the actual database backend.
"""

from lx_ai.integrations.endoreg_db.database.django_repository import (
    DjangoORMRepository,
)


class PostgresRepository(DjangoORMRepository):
    """Compatibility wrapper for existing PostgreSQL repository imports."""
