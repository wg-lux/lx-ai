"""Backward-compatible LX-AI media integration facade."""

from lx_ai.integrations.endoreg_db.media.legacy_bridge import (
    _field_file_is_encrypted,
    _path_is_encrypted,
    materialize_frames_for_lxai_annotations,
    plaintext_processed_video_path,
)

__all__ = [
    "_field_file_is_encrypted",
    "_path_is_encrypted",
    "materialize_frames_for_lxai_annotations",
    "plaintext_processed_video_path",
]
