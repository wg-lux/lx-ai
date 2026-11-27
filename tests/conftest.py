"""Pytest bootstrap for the endoreg-db runtime contract.

endoreg-db validates ``LX_RUNTIME_ROOT`` and its directory topology during
Django app startup and does not create it implicitly. Tests get an isolated,
explicitly provisioned runtime root so they never touch deployment data.
"""

from __future__ import annotations

import atexit
import os
import tempfile
from pathlib import Path

from endoreg_db.utils.file_operations import safe_rmtree
from endoreg_db.utils.paths import EndoregPathsModel

_RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="lx-ai-test-runtime-"))
os.environ["LX_RUNTIME_ROOT"] = str(_RUNTIME_ROOT)
EndoregPathsModel.from_root(_RUNTIME_ROOT).ensure_directories()
atexit.register(safe_rmtree, _RUNTIME_ROOT, missing_ok=True)
