from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime, parse_date
from endoreg_db.models import (
    AIDataSet,
    VideoFile,
    Frame,
    ImageClassificationAnnotation,
    Center,
    EndoscopyProcessor,
    InformationSource,
    ModelMeta,
    SensitiveMeta,
    VideoMeta,
    VideoState,
    VideoImportMeta,
    Patient,
    PatientExamination,
)


CSV_DIR = Path("/home/admin/csv")
ANNOTATION_CSV = CSV_DIR / "01_annotation.csv"
FRAME_CSV = CSV_DIR / "02_frame.csv"
VIDEO_CSV = CSV_DIR / "03_videofile.csv"


def _as_str_or_none(value: str | None) -> Optional[str]:
    if value is None:
        return None
    s = value.strip()
    return s if s != "" else None


def _as_int_or_none(value: str | None) -> Optional[int]:
    s = _as_str_or_none(value)
    if s is None:
        return None
    return int(s)


def _as_float_or_none(value: str | None) -> Optional[float]:
    s = _as_str_or_none(value)
    if s is None:
        return None
    return float(s)


def _as_bool(value: str | None, default: bool = False) -> bool:
    s = _as_str_or_none(value)
    if s is None:
        return default
    return s.lower() in {"1", "true", "t", "yes", "y"}


def _as_datetime_or_now(value: str | None):
    s = _as_str_or_none(value)
    if s is None:
        return timezone.now()
    parsed = parse_datetime(s)
    return parsed if parsed is not None else timezone.now()


def _as_date_or_none(value: str | None):
    s = _as_str_or_none(value)
    if s is None:
        return None
    return parse_date(s)


def _as_json_text(value: str | None, default_obj):
    s = _as_str_or_none(value)
    if s is None:
        return json.dumps(default_obj)
    try:
        parsed = json.loads(s)
        return json.dumps(parsed)
    except Exception:
        return json.dumps(default_obj)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV file: {path}")

def _fk_if_exists(Model, raw_value: str | None) -> Optional[int]:
    value = _as_int_or_none(raw_value)
    if value is None:
        return None
    return value if Model.objects.filter(id=value).exists() else None


def _required_fk_with_fallback(Model, raw_value: str | None, fallback_id: int) -> int:
    value = _as_int_or_none(raw_value)
    if value is not None and Model.objects.filter(id=value).exists():
        return value
    return fallback_id

@transaction.atomic
def run() -> None:
    print("=" * 80)
    print("START SQLITE CSV IMPORT")
    print("=" * 80)

    _require_file(VIDEO_CSV)
    _require_file(FRAME_CSV)
    _require_file(ANNOTATION_CSV)

    # -----------------------------------------------------------------
    # 1) Create new dataset
    # -----------------------------------------------------------------
    dataset = AIDataSet.objects.create(
        name=f"csv_import_{timezone.now().strftime('%Y%m%d_%H%M%S')}",
        description="Imported from CSV files in /home/admin/csv",
        ai_model_type="image_multilabel_classification",
        dataset_type="image",
        created_at=timezone.now(),
        updated_at=timezone.now(),
        is_active=True,
    )
    print(f"[DATASET] Created AIDataSet id={dataset.id}")

    # -----------------------------------------------------------------
    # 2) Import videofile rows
    # old CSV video id -> new DB video id
    # -----------------------------------------------------------------
    video_id_map: dict[int, int] = {}

    with VIDEO_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            old_video_id = int(row["id"])

            video = VideoFile.objects.create(
                raw_file=_as_str_or_none(row.get("raw_file")),
                processed_file=_as_str_or_none(row.get("processed_file")),
                video_hash=row["video_hash"].strip(),
                processed_video_hash=_as_str_or_none(row.get("processed_video_hash")),
                original_file_name=_as_str_or_none(row.get("original_file_name")),
                uploaded_at=_as_datetime_or_now(row.get("uploaded_at")),
                frame_dir=row["frame_dir"].strip(),
                fps=_as_float_or_none(row.get("fps")),
                duration=_as_float_or_none(row.get("duration")),
                frame_count=_as_int_or_none(row.get("frame_count")),
                width=_as_int_or_none(row.get("width")),
                height=_as_int_or_none(row.get("height")),
                suffix=_as_str_or_none(row.get("suffix")),
                sequences=_as_json_text(row.get("sequences"), {}),
                date=_as_date_or_none(row.get("date")),
                meta=_as_json_text(row.get("meta"), None) if _as_str_or_none(row.get("meta")) else None,
                date_created=_as_datetime_or_now(row.get("date_created")),
                date_modified=_as_datetime_or_now(row.get("date_modified")),
                ai_model_meta_id=_fk_if_exists(ModelMeta, row.get("ai_model_meta_id")),
                center_id=_required_fk_with_fallback(Center, row.get("center_id"), fallback_id=1),
                examination_id=_fk_if_exists(PatientExamination, row.get("examination_id")),
                patient_id=_fk_if_exists(Patient, row.get("patient_id")),
                processor_id=_fk_if_exists(EndoscopyProcessor, row.get("processor_id")),
                sensitive_meta_id=_fk_if_exists(SensitiveMeta, row.get("sensitive_meta_id")),
                import_meta_id=_fk_if_exists(VideoImportMeta, row.get("import_meta_id")),
                video_meta_id=_fk_if_exists(VideoMeta, row.get("video_meta_id")),
                state_id=_fk_if_exists(VideoState, row.get("state_id")),
                export_segments_by_video=_as_bool(row.get("export_segments_by_video"), default=False),
                uuid=row["uuid"].replace("-", "").strip(),
            )

            video_id_map[old_video_id] = int(video.id)

    print(f"[VIDEOFILE] Imported {len(video_id_map)} rows")

    # -----------------------------------------------------------------
    # 3) Import frame rows with remapped video_id
    # old CSV frame id -> new DB frame id
    # -----------------------------------------------------------------
    frame_id_map: dict[int, int] = {}

    with FRAME_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            old_frame_id = int(row["id"])
            old_video_id = int(row["video_id"])

            if old_video_id not in video_id_map:
                raise ValueError(
                    f"Frame {old_frame_id} references missing CSV video_id={old_video_id}"
                )

            frame = Frame.objects.create(
                frame_number=int(row["frame_number"]),
                relative_path=row["relative_path"].strip(),
                timestamp=_as_float_or_none(row.get("timestamp")),
                #old_examination_id=_as_int_or_none(row.get("old_examination_id")),
                is_extracted=_as_bool(row.get("is_extracted"), default=False),
                video_id=video_id_map[old_video_id],
            )

            frame_id_map[old_frame_id] = int(frame.id)

    print(f"[FRAME] Imported {len(frame_id_map)} rows")

    # -----------------------------------------------------------------
    # 4) Import annotation rows with remapped frame_id
    # also keep label_id as-is, exactly as you wanted
    # -----------------------------------------------------------------
    annotation_ids: list[int] = []

    with ANNOTATION_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            old_frame_id = int(row["frame_id"])

            if old_frame_id not in frame_id_map:
                raise ValueError(
                    f"Annotation row references missing CSV frame_id={old_frame_id}"
                )

            annotation = ImageClassificationAnnotation.objects.create(
                value=_as_bool(row.get("value"), default=False),
                float_value=_as_float_or_none(row.get("float_value")),
                annotator=_as_str_or_none(row.get("annotator")),
                date_created=_as_datetime_or_now(row.get("date_created")),
                date_modified=_as_datetime_or_now(row.get("date_modified")),
                frame_id=frame_id_map[old_frame_id],
                information_source_id=_fk_if_exists(InformationSource, row.get("information_source_id")),
                label_id=int(row["label_id"]),
                model_meta_id=_fk_if_exists(ModelMeta, row.get("model_meta_id")),
            )
            annotation_ids.append(int(annotation.id))

    print(f"[ANNOTATION] Imported {len(annotation_ids)} rows")

    # -----------------------------------------------------------------
    # 5) Create pivot rows through M2M
    # this inserts into endoreg_db_aidataset_image_annotations
    # -----------------------------------------------------------------
    dataset.image_annotations.add(*annotation_ids)
    dataset.updated_at = timezone.now()
    dataset.save(update_fields=["updated_at"])

    print(f"[PIVOT] Linked {len(annotation_ids)} annotations to dataset id={dataset.id}")

    # -----------------------------------------------------------------
    # 6) Validation summary
    # -----------------------------------------------------------------
    linked_annotation_count = dataset.image_annotations.count()

    imported_frame_count = (
        Frame.objects.filter(
            image_classification_annotations__in=dataset.image_annotations.all()
        )
        .distinct()
        .count()
    )

    imported_video_count = (
        VideoFile.objects.filter(
            frames__image_classification_annotations__in=dataset.image_annotations.all()
        )
        .distinct()
        .count()
    )

    print("-" * 80)
    print("[VALIDATION]")
    print(f"  Dataset ID           : {dataset.id}")
    print(f"  Linked annotations   : {linked_annotation_count}")
    print(f"  Distinct frames      : {imported_frame_count}")
    print(f"  Distinct videos      : {imported_video_count}")
    print("-" * 80)

    print("=" * 80)
    print("SQLITE CSV IMPORT COMPLETE")
    print("=" * 80)