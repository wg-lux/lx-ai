from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_training_frames_available(
    annotations: list[dict[str, Any]],
    *,
    output_root: Path | str,
    fps: float | None = 50.0,
    ext: str = "jpg",
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    annotation_ids = [
        int(ann["annotation_id"])
        for ann in annotations
        if ann.get("annotation_id") is not None
    ]

    if not annotation_ids:
        return annotations

    print(f"[FRAME MATERIALIZATION] annotations={len(annotation_ids)}")
    print(
        f"[FRAME MATERIALIZATION] output_root={Path(output_root).expanduser().resolve()}"
    )
    print(f"[FRAME MATERIALIZATION] fps={fps}, ext={ext}, overwrite={overwrite}")

    from lx_ai.utils.endoregdb_encrypted_frame_bridge import (
        materialize_frames_for_lxai_annotations,
    )

    # ------------------------------------------------------------------
    # TEMPORARY DEBUG LIMIT
    #
    # Restrict training frame materialization for first service validation.
    # Remove before real production training.
    # ------------------------------------------------------------------
    annotation_ids = annotation_ids[:5]

    print(
        f"[LXAI DEBUG] limiting frame materialization to "
        f"{len(annotation_ids)} annotations",
        flush=True,
    )

    path_by_annotation_id = materialize_frames_for_lxai_annotations(
        annotation_ids=annotation_ids,
        output_root=output_root,
        fps=fps,
        ext=ext,
        overwrite=overwrite,
    )

    missing_annotation_ids = [
        annotation_id
        for annotation_id in annotation_ids
        if annotation_id not in path_by_annotation_id
    ]

    if missing_annotation_ids:
        preview = missing_annotation_ids[:20]
        raise FileNotFoundError(
            "Frame materialization did not produce resolved paths for "
            f"{len(missing_annotation_ids)} annotations. "
            f"First missing annotation IDs: {preview}. "
            "Refusing to fall back to legacy/remapped frame paths because that can "
            "create wrong image-label pairs."
        )

    for ann in annotations:
        annotation_id = ann.get("annotation_id")
        if annotation_id is None:
            continue

        materialized_path = path_by_annotation_id.get(int(annotation_id))
        if materialized_path is None:
            continue

        ann.setdefault("frame", {})
        ann["frame"]["resolved_frame_path"] = str(materialized_path)

    return annotations
