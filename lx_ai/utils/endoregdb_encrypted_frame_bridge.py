"""Backward-compatible LX-AI media integration facade."""

from lx_ai.integrations.endoreg_db.media.legacy_bridge import (
    plaintext_processed_video_path,
    materialize_frames_for_lxai_annotations,
)

__all__ = [
    "plaintext_processed_video_path",
    "materialize_frames_for_lxai_annotations",
]
