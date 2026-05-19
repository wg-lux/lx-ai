from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from django.db.models import QuerySet

from endoreg_db.export.frames.export_frames_with_labels import (
    _assert_video_media_export_ready,
    _extract_and_move_transcoded_frames,
    _frame_pk_filename,
    _resolve_processed_video_source_path,
)
from endoreg_db.models import ImageClassificationAnnotation, VideoFile
from endoreg_db.utils.encryption.encryption import MAGIC
from endoreg_db.utils.encryption.storage_materialization import (
    materialized_plaintext_field_file,
)
from endoreg_db.utils.file_operations import ensure_directory


def _path_is_encrypted(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


def _field_file_is_encrypted(video: VideoFile, source_path: Path | None) -> bool:
    processed_file = getattr(video, "processed_file", None)
    storage = getattr(processed_file, "storage", None)

    if processed_file and storage and hasattr(storage, "is_encrypted"):
        try:
            return bool(storage.is_encrypted(processed_file.name))
        except Exception:
            pass

    return bool(
        source_path and source_path.exists() and _path_is_encrypted(source_path)
    )


@contextmanager
def plaintext_processed_video_path(video: VideoFile) -> Iterator[Path]:
    """
    Yield an FFmpeg-readable processed video path.

    - Plaintext local files are used directly.
    - Encrypted endoreg-db files are decrypted into a temporary plaintext file.
    - Temporary plaintext is deleted automatically.
    """
    processed_file = getattr(video, "processed_file", None)
    if not processed_file or not getattr(processed_file, "name", None):
        raise FileNotFoundError(
            f"processed video artifact missing for video={video.pk}"
        )

    source_path = _resolve_processed_video_source_path(video)

    if (
        source_path
        and source_path.exists()
        and not _field_file_is_encrypted(video, source_path)
    ):
        yield source_path
        return

    suffix = Path(str(processed_file.name)).suffix or ".mp4"

    with materialized_plaintext_field_file(
        processed_file,
        suffix=suffix,
        prefix=f"lx-ai-video-{video.pk}-",
    ) as tmp_plaintext_path:
        yield tmp_plaintext_path


def materialize_frames_for_lxai_annotations(
    *,
    annotation_ids: list[int],
    output_root: Path | str,
    fps: float | None,
    ext: str = "jpg",
    quality: int = 2,
    overwrite: bool = False,
) -> dict[int, str]:
    """
    lx-ai owned frame materialization bridge.

    Keeps endoreg-db unchanged but reuses:
    - endoreg-db ORM models
    - endoreg-db export readiness checks
    - endoreg-db encrypted storage decryption helper
    - endoreg-db FFmpeg frame extraction/mapping helper
    """
    root = ensure_directory(Path(output_root))

    annotations: QuerySet[ImageClassificationAnnotation] = (
        ImageClassificationAnnotation.objects.filter(pk__in=annotation_ids)
        .select_related("frame", "frame__video")
        .order_by("frame__video_id", "frame_id", "id")
    )

    video_frame_pks: dict[int, set[int]] = {}
    annotation_to_frame_pk: dict[int, int] = {}

    for ann in annotations.iterator(chunk_size=2000):
        if ann.frame_id is None or ann.frame is None or ann.frame.video_id is None:
            continue

        video_id = int(ann.frame.video_id)
        frame_pk = int(ann.frame_id)

        video_frame_pks.setdefault(video_id, set()).add(frame_pk)
        annotation_to_frame_pk[int(ann.pk)] = frame_pk

    if not video_frame_pks:
        return {}

    for video in VideoFile.objects.filter(pk__in=video_frame_pks.keys()).order_by("pk"):
        _assert_video_media_export_ready(video)

        frame_dir = ensure_directory(root / f"video_{video.pk}")
        # tmp_dir = ensure_directory(frame_dir / f"lx_ai_transcode_tmp_{uuid.uuid4().hex}")

        with plaintext_processed_video_path(video) as source_path:
            _extract_and_move_transcoded_frames(
                video,
                source_path=source_path,
                frame_dir=frame_dir,
                frame_pks=video_frame_pks.get(int(video.pk)),
                fps=fps,
                quality=quality,
                ext=ext,
                overwrite=overwrite,
            )

    out: dict[int, str] = {}
    for annotation_id, frame_pk in annotation_to_frame_pk.items():
        for video_id, frame_pks in video_frame_pks.items():
            if frame_pk in frame_pks:
                path = root / f"video_{video_id}" / _frame_pk_filename(frame_pk, ext)
                if path.exists():
                    out[annotation_id] = str(path)
                break

    return out
