from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.xfail(
    reason=(
        "Pending application fix: dataset loader must prefer resolved/materialized "
        "frame paths over stale DB frame paths."
    ),
    strict=False,
)
def test_dataset_loader_prefers_resolved_frame_path_when_present() -> None:
    """
    Regression contract for the gc-10 production failure.

    The materializer created/validated frames under:
      /var/endoreg-service-user/lx-ai/data/frames/generated/...

    But the dataset loader later tried stale DB paths under:
      /var/endoreg-service-user/lx-annotate/data/frames/...

    Desired rule:
      resolved_frame_path wins if present.
      frame_path is only fallback.
    """
    source_file = Path("lx_ai/utils/data_loader_for_model_training.py")
    assert source_file.exists(), "data_loader_for_model_training.py must exist"

    source = source_file.read_text(encoding="utf-8")

    assert "resolved_frame_path" in source

    resolved_idx = source.find("resolved_frame_path")
    frame_path_idx = source.find("frame_path")

    assert resolved_idx != -1
    assert frame_path_idx != -1
    assert resolved_idx < frame_path_idx
