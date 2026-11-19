"""PostgreSQL implementation of the EndoRegDB repository."""

import psycopg

from lx_ai.integrations.endoreg_db.database.legacy_sql import (
    load_annotations_from_postgres,
    load_labelset_from_postgres,
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


class PostgresRepository:
    """Concrete PostgreSQL repository."""

    def get_annotations(
        self,
        dataset_id: int,
    ) -> list[AnnotationDTO]:
        try:
            rows = load_annotations_from_postgres(dataset_id)

        except (psycopg.Error, OSError, RuntimeError) as exc:
            raise EndoRegDBConnectionError(
                f"Could not retrieve annotations for dataset "
                f"{dataset_id} from PostgreSQL."
            ) from exc

        return annotations_to_dto(rows)

    def get_labelset(
        self,
        labelset_id: int,
        labelset_version: int,
    ) -> LabelSetDTO:
        try:
            data = load_labelset_from_postgres(
                labelset_id=labelset_id,
                labelset_version=labelset_version,
            )

        except ValueError as exc:
            raise EndoRegDBDataError(
                f"Labelset {labelset_id}, version "
                f"{labelset_version}, could not be retrieved."
            ) from exc

        except (psycopg.Error, OSError, RuntimeError) as exc:
            raise EndoRegDBConnectionError(
                "Could not access PostgreSQL labelset data."
            ) from exc

        return labelset_to_dto(data)
