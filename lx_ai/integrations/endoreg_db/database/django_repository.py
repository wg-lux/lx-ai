"""Django ORM implementation of the LX-AI EndoRegDB repository.

All EndoRegDB database access is isolated in this module.

LX-AI's training pipeline consumes only normalized DTOs and does not
depend on EndoRegDB's database tables or SQL schema.
"""

from __future__ import annotations

from typing import Any

from django.db import DatabaseError

from lx_ai.integrations.endoreg_db.database.mappers import (
    annotations_to_dto,
    labelset_to_dto,
)
from lx_ai.integrations.endoreg_db.dto import (
    AnnotationDTO,
    LabelSetDTO,
)
from lx_ai.integrations.endoreg_db.exceptions import (
    EndoRegDBCompatibilityError,
    EndoRegDBConnectionError,
    EndoRegDBDataError,
)


class DjangoORMRepository:
    """Retrieve EndoRegDB training data using Django ORM."""

    @staticmethod
    def _get_endoreg_models() -> tuple[Any, Any]:
        """Import models lazily after Django initialization."""

        try:
            from endoreg_db.models import AIDataSet, LabelSet

        except ImportError as exc:
            raise EndoRegDBCompatibilityError(
                "Required EndoRegDB models AIDataSet and LabelSet "
                "are not available in the installed EndoRegDB version."
            ) from exc

        return AIDataSet, LabelSet

    def get_annotations(
        self,
        dataset_id: int,
    ) -> list[AnnotationDTO]:
        """Retrieve frame annotations belonging to an AI dataset."""

        AIDataSet, _ = self._get_endoreg_models()

        try:
            dataset = AIDataSet.objects.filter(pk=dataset_id).first()

            # Preserve the original SQL behavior:
            # a dataset with no matching annotations returns [].
            if dataset is None:
                return []

            queryset = (
                dataset.get_image_annotations_queryset()
                .filter(
                    frame__isnull=False,
                    frame__video__isnull=False,
                    label__isnull=False,
                )
                .select_related(
                    "frame",
                    "frame__video",
                    "label",
                )
            )
            rows: list[dict[str, Any]] = []

            for annotation in queryset.iterator(chunk_size=2000):
                frame = annotation.frame
                video = frame.video
                label = annotation.label

                processed_file = getattr(
                    video,
                    "processed_file",
                    None,
                )

                rows.append(
                    {
                        "dataset_id": dataset.pk,
                        "annotation_id": annotation.pk,
                        "annotation": {
                            "id": annotation.pk,
                        },
                        "frame": {
                            "id": frame.pk,
                            "relative_path": frame.relative_path,
                            "frame_number": frame.frame_number,
                            "timestamp": frame.timestamp,
                            # Preserve the original SQL output.
                            "file_path": video.frame_dir,
                            "frame_dir": video.frame_dir,
                            # This field is not present in every
                            # EndoRegDB model version.
                            "old_examination_id": getattr(
                                frame,
                                "old_examination_id",
                                None,
                            ),
                            "video_id": video.pk,
                            "video_uuid": (
                                str(video.uuid)
                                if getattr(video, "uuid", None) is not None
                                else None
                            ),
                            "processed_file": getattr(
                                processed_file,
                                "name",
                                None,
                            ),
                            "video_fps": video.fps,
                        },
                        "label": {
                            "id": label.pk,
                            "name": label.name,
                        },
                        "value": annotation.value,
                        "annotator": annotation.annotator,
                    }
                )

            return annotations_to_dto(rows)

        except DatabaseError as exc:
            raise EndoRegDBConnectionError(
                f"Could not retrieve annotations "
                f"for dataset {dataset_id} using Django ORM."
            ) from exc

        except (AttributeError, TypeError) as exc:
            raise EndoRegDBCompatibilityError(
                "The installed EndoRegDB model structure is "
                "incompatible with the LX-AI annotation adapter."
            ) from exc

    def get_labelset(
        self,
        labelset_id: int,
        labelset_version: int,
    ) -> LabelSetDTO:
        """Retrieve a labelset and preserve prototype label ordering."""

        _, LabelSet = self._get_endoreg_models()

        try:
            labelset = LabelSet.objects.filter(
                pk=labelset_id,
                version=labelset_version,
            ).first()

            if labelset is None:
                raise EndoRegDBDataError(
                    f"Labelset {labelset_id}, version "
                    f"{labelset_version}, could not be retrieved."
                )

            # IMPORTANT:
            # Prototype SQL used ORDER BY label.id.
            # Do not use get_labels_in_order(), because the
            # current EndoRegDB implementation sorts by name.

            # labels = labelset.labels.all().order_by("pk")
            labels = labelset.labels.all().order_by("id")

            label_data = [
                {
                    "id": label.pk,
                    "name": label.name,
                }
                for label in labels
            ]

            if not label_data:
                raise EndoRegDBDataError(
                    f"Labelset {labelset_id}, version "
                    f"{labelset_version}, contains no labels."
                )

            return labelset_to_dto(
                {
                    "id": labelset.pk,
                    "version": labelset.version,
                    "labels": label_data,
                }
            )

        except DatabaseError as exc:
            raise EndoRegDBConnectionError(
                "Could not retrieve EndoRegDB labelset "
                f"{labelset_id}, version {labelset_version}."
            ) from exc

        except (AttributeError, TypeError) as exc:
            raise EndoRegDBCompatibilityError(
                "The installed EndoRegDB labelset model "
                "is incompatible with the LX-AI adapter."
            ) from exc
