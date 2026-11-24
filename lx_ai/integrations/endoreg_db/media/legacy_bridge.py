from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from endoreg_db.utils.encryption.encrypted import MAGIC
from endoreg_db.utils.filesystem.file_operations import (
    ensure_directory,
    safe_unlink_file,
)


if TYPE_CHECKING:
    from endoreg_db.models import VideoFile

from endoreg_db.utils.storage_streaming import field_file_size, iter_field_file_bytes


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
        except (OSError, ValueError, NotImplementedError):
            # Fall back to checking the local file's encryption header.
            pass

    return bool(
        source_path and source_path.exists() and _path_is_encrypted(source_path)
    )


def _encryption_key_configured() -> bool:
    key_value = os.environ.get("LX_ANNOTATE_MASTER_KEY", "").strip()
    key_file = os.environ.get("LX_ANNOTATE_MASTER_KEY_FILE", "").strip()

    return bool(key_value or (key_file and Path(key_file).expanduser().is_file()))


def _require_encryption_key_for_video(video: VideoFile) -> None:
    if _encryption_key_configured():
        return

    raise RuntimeError(
        "VideoFile.processed_file appears to be encrypted, but no readable "
        "LX_ANNOTATE_MASTER_KEY or LX_ANNOTATE_MASTER_KEY_FILE is configured. "
        f"video_id={video.pk}. Local plaintext videos do not require this key, "
        "but encrypted videos and production lx-ai runs must use the same "
        "application master key that was used by lx-annotate/endoreg-db."
    )


def _plaintext_tmp_dir() -> Path | None:
    raw = os.environ.get("ENDOREG_DB_PLAINTEXT_TMP_DIR", "").strip()
    if not raw:
        return None

    return ensure_directory(Path(raw).expanduser().resolve())


@contextmanager
def _materialized_plaintext_field_file_for_lxai(
    field_file,
    *,
    suffix: str = "",
    prefix: str = "lx-ai-fieldfile-",
) -> Iterator[Path]:
    tmp_root = _plaintext_tmp_dir()
    size = field_file_size(field_file)

    with NamedTemporaryFile(
        mode="wb",
        prefix=prefix,
        suffix=suffix,
        dir=str(tmp_root) if tmp_root is not None else None,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        for chunk in iter_field_file_bytes(field_file, start=0, end=size - 1):
            tmp.write(chunk)

    try:
        yield tmp_path
    finally:
        safe_unlink_file(tmp_path, missing_ok=True)


@contextmanager
def plaintext_processed_video_path(video: VideoFile) -> Iterator[Path]:
    """Yield an FFmpeg-readable processed video path.

    Plaintext local files are reused.
    Encrypted files are materialized through EndoRegDB storage streaming.
    Temporary plaintext files are cleaned up automatically.
    """

    processed_file = getattr(video, "processed_file", None)

    if not processed_file or not getattr(processed_file, "name", None):
        raise FileNotFoundError(
            f"processed video artifact missing for video={video.pk}"
        )

    from endoreg_db.export.frames.export_frames_with_labels import (
        _resolve_processed_video_source_path,
    )

    source_path = _resolve_processed_video_source_path(video)

    # Do not send an encrypted local file directly to FFmpeg.
    source_is_encrypted = _field_file_is_encrypted(video, source_path)

    # Reuse only an existing plaintext file.
    if source_path is not None and source_path.exists() and not source_is_encrypted:
        yield source_path
        return

    # Preserve the prototype's encryption-key requirement.
    if source_is_encrypted or source_path is None or not source_path.exists():
        _require_encryption_key_for_video(video)

    suffix = Path(str(processed_file.name)).suffix or ".mp4"

    # Preserve the configured plaintext temporary directory
    # and the original automatic cleanup behavior.
    with _materialized_plaintext_field_file_for_lxai(
        processed_file,
        suffix=suffix,
        prefix=f"lx-ai-video-{video.pk}-",
    ) as plaintext_path:
        # Fail safely if EndoRegDB did not return decrypted bytes.
        if _path_is_encrypted(plaintext_path):
            raise RuntimeError(
                "EndoRegDB returned encrypted video content instead "
                f"of plaintext for video_id={video.pk}."
            )

        yield plaintext_path


def materialize_frames_for_lxai_annotations(
    *,
    annotation_ids: list[int],
    output_root: Path | str,
    fps: float | None,
    ext: str = "jpg",
    quality: int = 2,
    overwrite: bool = False,
) -> dict[int, str]:
    from endoreg_db.export.frames.export_frames_with_labels import (
        _assert_video_media_export_ready,
        _extract_and_move_transcoded_frames,
        _frame_pk_filename,
    )
    from endoreg_db.models import ImageClassificationAnnotation, VideoFile

    """
    lx-ai owned frame materialization bridge.

    Keeps endoreg-db unchanged but reuses:
    - endoreg-db ORM models
    - endoreg-db export readiness checks
    - endoreg-db encrypted storage decryption helper
    - endoreg-db FFmpeg frame extraction/mapping helper
    """
    root = ensure_directory(Path(output_root))

    annotations = (
        ImageClassificationAnnotation.objects.filter(pk__in=annotation_ids)
        .select_related("frame", "frame__video")
        .order_by("frame__video_id", "frame_id", "id")
    )

    video_frame_pks: dict[int, set[int]] = {}
    annotation_to_frame: dict[int, tuple[int, int]] = {}

    for ann in annotations.iterator(chunk_size=2000):
        if ann.frame_id is None or ann.frame is None or ann.frame.video_id is None:
            continue

        video_id = int(ann.frame.video_id)
        frame_pk = int(ann.frame_id)

        video_frame_pks.setdefault(video_id, set()).add(frame_pk)
        annotation_to_frame[int(ann.pk)] = (video_id, frame_pk)

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

            expected_frame_pks = video_frame_pks.get(int(video.pk), set())
            missing_frame_paths = [
                frame_dir / _frame_pk_filename(frame_pk, ext)
                for frame_pk in sorted(expected_frame_pks)
                if not (frame_dir / _frame_pk_filename(frame_pk, ext)).exists()
            ]

            if missing_frame_paths:
                preview = [str(path) for path in missing_frame_paths[:20]]
                raise FileNotFoundError(
                    f"Frame materialization failed for video={video.pk}. "
                    f"Missing {len(missing_frame_paths)} generated frames after "
                    f"endoreg-db extraction fallback. First missing: {preview}"
                )

    out: dict[int, str] = {}

    for annotation_id, (video_id, frame_pk) in annotation_to_frame.items():
        path = root / f"video_{video_id}" / _frame_pk_filename(frame_pk, ext)

        if path.is_file():
            out[annotation_id] = str(path)

    return out
