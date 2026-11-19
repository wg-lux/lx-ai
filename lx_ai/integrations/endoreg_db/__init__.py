"""EndoRegDB integration boundary for LX-AI."""

from lx_ai.integrations.endoreg_db.contracts import (
    DatabaseRepository,
    DatasetRepository,
    FrameMaterializer,
    LabelSetRepository,
    ProcessedVideoProvider,
)
from lx_ai.integrations.endoreg_db.exceptions import (
    EndoRegDBCompatibilityError,
    EndoRegDBConnectionError,
    EndoRegDBDataError,
    EndoRegDBIntegrationError,
    EndoRegDBMediaError,
)

__all__ = [
    "DatabaseRepository",
    "DatasetRepository",
    "LabelSetRepository",
    "FrameMaterializer",
    "ProcessedVideoProvider",
    "EndoRegDBIntegrationError",
    "EndoRegDBConnectionError",
    "EndoRegDBDataError",
    "EndoRegDBMediaError",
    "EndoRegDBCompatibilityError",
]
