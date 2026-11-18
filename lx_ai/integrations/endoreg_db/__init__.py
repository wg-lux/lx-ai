"""
EndoRegDB integration boundary for LX-AI.

All EndoRegDB-specific access should be isolated inside this package.

The rest of LX-AI should consume LX-AI-owned data contracts rather
than importing EndoRegDB models or accessing its database schema.
"""

from lx_ai.integrations.endoreg_db.exceptions import (
    EndoRegDBIntegrationError,
    EndoRegDBConnectionError,
    EndoRegDBDataError,
    EndoRegDBMediaError,
)

__all__ = [
    "EndoRegDBIntegrationError",
    "EndoRegDBConnectionError",
    "EndoRegDBDataError",
    "EndoRegDBMediaError",
]