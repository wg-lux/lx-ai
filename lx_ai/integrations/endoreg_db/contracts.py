
"""Stable LX-AI interfaces for external data access."""

from typing import Protocol

from lx_ai.integrations.endoreg_db.dto import (
    AnnotationDTO,
    LabelSetDTO,
)


class DatasetRepository(Protocol):
    """Dataset annotation retrieval contract."""

    def get_annotations(
        self,
        dataset_id: int,
    ) -> list[AnnotationDTO]:
        ...


class LabelSetRepository(Protocol):
    """Labelset retrieval contract."""

    def get_labelset(
        self,
        labelset_id: int,
        labelset_version: int,
    ) -> LabelSetDTO:
        ...
