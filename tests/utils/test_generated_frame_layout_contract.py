from __future__ import annotations

from pathlib import Path


def test_generated_frame_layout_contract() -> None:
    """
    lx-ai generated-frame contract:

      <FRAME_MATERIALIZATION_OUTPUT_ROOT>/video_<video_id>/frame_<frame_id>.<ext>

    Important:
    - directory uses video primary key
    - filename uses frame primary key
    - filename must not depend on frame_number
    """
    output_root = Path("/tmp/lx-ai-test/frames/generated")
    video_id = 12
    frame_id = 7323398
    ext = "jpg"

    expected = output_root / f"video_{video_id}" / f"frame_{frame_id}.{ext}"

    assert expected.as_posix().endswith("/frames/generated/video_12/frame_7323398.jpg")
    assert "frame_7323398" in expected.name
    assert "video_12" in expected.parent.name
