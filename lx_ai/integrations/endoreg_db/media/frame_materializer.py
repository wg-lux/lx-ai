"""Concrete EndoRegDB media integration adapter."""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from lx_ai.integrations.endoreg_db.media.legacy_bridge import (
    materialize_frames_for_lxai_annotations,
    plaintext_processed_video_path,
)


class EndoRegDBMediaAdapter:
    """Adapter for EndoRegDB video and frame operations."""

    def materialize_frames(
        self,
        *,
        annotation_ids: list[int],
        output_root: Path | str,
        fps: float | None,
        ext: str = "jpg",
        quality: int = 2,
        overwrite: bool = False,
    ) -> dict[int, str]:
        return materialize_frames_for_lxai_annotations(
            annotation_ids=annotation_ids,
            output_root=output_root,
            fps=fps,
            ext=ext,
            quality=quality,
            overwrite=overwrite,
        )

    def plaintext_processed_video_path(
        self,
        video: Any,
    ) -> AbstractContextManager[Path]:
        return plaintext_processed_video_path(video)
