
"""Encryption and plaintext materialization adapter for EndoRegDB.

LX-AI delegates all encryption and decryption operations to EndoRegDB.
No encryption format, key handling, or cryptographic implementation
is duplicated here.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from endoreg_db.utils.storage import ensure_local_file


@contextmanager
def plaintext_field_file_path(
    field_file,
    *,
    suffix: str | None = None,
) -> Iterator[Path]:
    """Yield a local plaintext path for an EndoRegDB FieldFile.

    EndoRegDB owns:
    - Encryption detection
    - Master-key resolution
    - Decryption
    - Storage access
    - Temporary file creation
    - Temporary file cleanup

    Plaintext local files are reused where possible.
    Encrypted files are transparently decrypted by EndoRegDB storage.
    """
    if not field_file or not getattr(field_file, "name", None):
        raise FileNotFoundError(
            "Cannot materialize an empty EndoRegDB FieldFile."
        )

    try:
        with ensure_local_file(field_file, suffix=suffix) as local_path:
            yield Path(local_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not materialize EndoRegDB file "
            f"{field_file.name!r} as plaintext for LX-AI."
        ) from exc
