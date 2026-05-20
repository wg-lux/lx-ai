from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_training_frames_available(
    annotations: list[dict[str, Any]],
    *,
    output_root: Path | str,
    fps: float | None = 50.0,
    ext: str = "jpg",
    overwrite: bool = False,
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

    path_by_annotation_id = materialize_frames_for_lxai_annotations(
        annotation_ids=annotation_ids,
        output_root=output_root,
        fps=fps,
        ext=ext,
        overwrite=overwrite,
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
