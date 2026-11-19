"""Backward-compatible facade for LX-AI database loading.

All database-specific implementation lives inside the
EndoRegDB integration adapter.

Existing callers retain their original import paths.
"""

from lx_ai.integrations.endoreg_db.database.legacy_sql import (
    load_annotations,
    load_annotations_from_postgres,
    load_annotations_from_sqlite,
    load_labelset_from_postgres,
    load_labelset_from_sqlite,
)

__all__ = [
    "load_annotations",
    "load_annotations_from_postgres",
    "load_annotations_from_sqlite",
    "load_labelset_from_postgres",
    "load_labelset_from_sqlite",
]
