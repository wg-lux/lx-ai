"""SQLite implementation of the EndoRegDB repository."""

import sqlite3

from lx_ai.integrations.endoreg_db.database.legacy_sql import (
    load_annotations_from_sqlite,
    load_labelset_from_sqlite,
)
from lx_ai.integrations.endoreg_db.database.mappers import (
    annotations_to_dto,
    labelset_to_dto,
)
from lx_ai.integrations.endoreg_db.dto import (
    AnnotationDTO,
    LabelSetDTO,
)
from lx_ai.integrations.endoreg_db.exceptions import (
    EndoRegDBConnectionError,
    EndoRegDBDataError,
)


class SQLiteRepository:
    """Concrete SQLite repository."""

    def get_annotations(
        self,
        dataset_id: int,
    ) -> list[AnnotationDTO]:
        try:
            rows = load_annotations_from_sqlite(dataset_id)

        except (sqlite3.Error, OSError) as exc:
            raise EndoRegDBConnectionError(
                f"Could not retrieve annotations for dataset {dataset_id} from SQLite."
            ) from exc

        return annotations_to_dto(rows)

    def get_labelset(
        self,
        labelset_id: int,
        labelset_version: int,
    ) -> LabelSetDTO:
        try:
            data = load_labelset_from_sqlite(
                labelset_id=labelset_id,
                labelset_version=labelset_version,
            )

        except ValueError as exc:
            raise EndoRegDBDataError(
                f"Labelset {labelset_id}, version "
                f"{labelset_version}, could not be retrieved."
            ) from exc

        except (sqlite3.Error, OSError) as exc:
            raise EndoRegDBConnectionError(
                "Could not access SQLite labelset data."
            ) from exc

        return labelset_to_dto(data)
