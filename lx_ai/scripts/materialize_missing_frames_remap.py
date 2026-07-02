from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image
from endoreg_db.models import Frame

import os

SOURCE_ROOT = Path(os.getenv("FRAME_PATH_REMAP_SOURCE", "")).expanduser()
TARGET_ROOT = Path(os.getenv("FRAME_PATH_REMAP_TARGET", "")).expanduser()
# SOURCE_ROOT = Path("")
# TARGET_ROOT = Path("")
PLACEHOLDER_SIZE = (224, 224)


def _iter_target_paths() -> Iterable[Path]:
    for frame in Frame.objects.select_related("video").iterator():
        frame_dir = getattr(frame.video, "frame_dir", None)
        relative_path = getattr(frame, "relative_path", None)

        if not frame_dir or not relative_path:
            continue

        frame_dir_path = Path(str(frame_dir))

        try:
            relative_dir = frame_dir_path.relative_to(SOURCE_ROOT)
        except ValueError:
            continue

        yield TARGET_ROOT / relative_dir / str(relative_path)


def _write_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", PLACEHOLDER_SIZE, color=(0, 0, 0))
    img.save(path, format="JPEG", quality=90)


def run() -> None:
    total = 0
    created = 0
    existing = 0
    skipped = 0

    if not str(SOURCE_ROOT) or not str(TARGET_ROOT):
        raise RuntimeError(
            "FRAME_PATH_REMAP_SOURCE and FRAME_PATH_REMAP_TARGET must be set"
        )

    for target_path in _iter_target_paths():
        total += 1

        if target_path.is_file():
            existing += 1
            continue

        try:
            _write_placeholder(target_path)
            created += 1
        except Exception:
            skipped += 1

    print("=" * 80)
    print("MATERIALIZE REMAPPED PLACEHOLDER FRAMES")
    print("=" * 80)
    print(f"Total candidate paths : {total}")
    print(f"Already existing      : {existing}")
    print(f"Created placeholders  : {created}")
    print(f"Skipped               : {skipped}")
    print(f"Target root           : {TARGET_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    run()


#
# secretspec run --provider env -- uv run python manage.py shell -c "from lx_ai.scripts.materialize_missing_frames_remap import run; run()"
