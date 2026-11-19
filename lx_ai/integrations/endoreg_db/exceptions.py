"""Exceptions for the LX-AI EndoRegDB integration."""


class EndoRegDBIntegrationError(RuntimeError):
    """Base exception for EndoRegDB integration failures."""


class EndoRegDBConnectionError(EndoRegDBIntegrationError):
    """Raised when the external database cannot be accessed."""


class EndoRegDBDataError(EndoRegDBIntegrationError):
    """Raised when external data is missing or invalid."""


class EndoRegDBMediaError(EndoRegDBIntegrationError):
    """Raised when external media cannot be resolved or accessed."""


class EndoRegDBCompatibilityError(EndoRegDBIntegrationError):
    """Raised when the external integration API is incompatible."""
