from __future__ import annotations

from pathlib import Path


def test_temporary_frame_materialization_limit_is_visible_and_documented() -> None:
    """
    Temporary service-validation contract:

    First production service runs must not materialize the full training dataset
    accidentally. The temporary limit must be obvious in source and easy to remove.

    This test intentionally checks for the explicit debug guard/comment/log.
    """
    path = Path("lx_ai/utils/frame_materializer.py")
    assert path.exists(), "frame_materializer.py must exist"

    source = path.read_text(encoding="utf-8")

    assert "annotation_ids" in source
    assert "[:5]" in source or "LXAI_FRAME_MATERIALIZATION_LIMIT" in source
    assert "DEBUG" in source.upper() or "TEMPORARY" in source.upper()
    assert "limiting frame materialization" in source.lower()
