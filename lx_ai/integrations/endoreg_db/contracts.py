"""Stable LX-AI interfaces for external data access."""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from lx_ai.integrations.endoreg_db.dto import (
    AnnotationDTO,
    LabelSetDTO,
)


class DatasetRepository(Protocol):
    """Contract for retrieving dataset annotations."""

    def get_annotations(
        self,
        dataset_id: int,
    ) -> list[AnnotationDTO]: ...


class LabelSetRepository(Protocol):
    """Contract for retrieving a specific labelset version."""

    def get_labelset(
        self,
        labelset_id: int,
        labelset_version: int,
    ) -> LabelSetDTO: ...


class DatabaseRepository(
    DatasetRepository,
    LabelSetRepository,
    Protocol,
):
    """Complete database interface consumed by LX-AI."""

    ...


class FrameMaterializer(Protocol):
    """Contract for materializing annotation frames."""

    def materialize_frames(
        self,
        *,
        annotation_ids: list[int],
        output_root: Path | str,
        fps: float | None,
        ext: str = "jpg",
        quality: int = 2,
        overwrite: bool = False,
    ) -> dict[int, str]: ...


class ProcessedVideoProvider(Protocol):
    """Contract for obtaining plaintext processed videos."""

    def plaintext_processed_video_path(
        self,
        video: object,
    ) -> AbstractContextManager[Path]: ...
