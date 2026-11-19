"""Public media integration API."""

from lx_ai.integrations.endoreg_db.media.encryption_adapter import (
    plaintext_field_file_path,
)
from lx_ai.integrations.endoreg_db.media.frame_materializer import (
    EndoRegDBMediaAdapter,
)
from lx_ai.integrations.endoreg_db.media.legacy_bridge import (
    materialize_frames_for_lxai_annotations,
    plaintext_processed_video_path,
)

__all__ = [
    "EndoRegDBMediaAdapter",
    "plaintext_field_file_path",
    "plaintext_processed_video_path",
    "materialize_frames_for_lxai_annotations",
]
