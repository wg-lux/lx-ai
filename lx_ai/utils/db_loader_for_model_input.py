# lx_ai/utils/db_loader_for_model_input.py
from __future__ import annotations

import os
import psycopg
from typing import Any, Dict, List

def load_annotations_from_postgres(dataset_id: int) -> list[dict]:
    password = None
    pw_file = os.getenv("DEV_DB_PASSWORD_FILE")
    if pw_file:
        with open(pw_file, "r") as f:
            password = f.read().strip()

    sql = """
    SELECT
        f.id                    AS frame_id,
        f.relative_path         AS file_path,
        vf.frame_dir            AS frame_dir,
        f.old_examination_id    AS old_examination_id,
        l.id                    AS label_id,
        l.name                  AS label_name,
        a.value                 AS value
    FROM endoreg_db_aidataset_image_annotations dai
    JOIN endoreg_db_imageclassificationannotation a
        ON a.id = dai.imageclassificationannotation_id
    JOIN endoreg_db_frame f
        ON f.id = a.frame_id
    JOIN endoreg_db_videofile vf
        ON vf.id = f.video_id
    JOIN endoreg_db_label l
        ON l.id = a.label_id
    WHERE dai.aidataset_id = %s
    """

    rows: list[dict] = []

    #  CORRECT: cursor is created unconditionally
    with psycopg.connect(
        host=os.getenv("DEV_DB_HOST"),
        port=int(os.getenv("DEV_DB_PORT")),
        dbname=os.getenv("DEV_DB_NAME"),
        user=os.getenv("DEV_DB_USER"),
        password=password,
        sslmode="disable",
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (dataset_id,))
            for row in cur.fetchall():
                rows.append(
                    {
                        "frame": {
                            "id": row[0],
                            "relative_path": row[1],
                            "file_path": row[2],
                            "old_examination_id": row[3],
                        },
                        "label": {
                            "id": row[4],
                            "name": row[5],
                        },
                        "value": row[6],
                    }
                )

    return rows

def load_labelset_from_postgres(
    *,
    labelset_id: int,
    labelset_version: int,
) -> dict:
    """
    Load labelset metadata + ordered labels from Postgres.

    Mirrors endoreg-db semantics:
      - labelset.id
      - labelset.version
      - labels via endoreg_db_labelset_labels
    """

    password = None
    pw_file = os.getenv("DEV_DB_PASSWORD_FILE")
    if pw_file:
        with open(pw_file, "r") as f:
            password = f.read().strip()

    sql = """
    SELECT
        ls.id            AS labelset_id,
        ls.version       AS version,
        l.id             AS label_id,
        l.name           AS label_name
    FROM endoreg_db_labelset ls
    JOIN endoreg_db_labelset_labels lsl
        ON lsl.labelset_id = ls.id
    JOIN endoreg_db_label l
        ON l.id = lsl.label_id
    WHERE ls.id = %s
      AND ls.version = %s
    ORDER BY l.id
    """

    labels = []

    with psycopg.connect(
        host=os.getenv("DEV_DB_HOST"),
        port=int(os.getenv("DEV_DB_PORT")),
        dbname=os.getenv("DEV_DB_NAME"),
        user=os.getenv("DEV_DB_USER"),
        password=password,
        sslmode="disable",
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (labelset_id, labelset_version))
            rows = cur.fetchall()

    if not rows:
        raise ValueError(
            f"No labels found for labelset_id={labelset_id}, version={labelset_version}"
        )

    for row in rows:
        labels.append(
            {
                "id": row[2],
                "name": row[3],
            }
        )

    return {
        "id": labelset_id,
        "version": labelset_version,
        "labels": labels,
    }





def _get_password() -> str:
    pw = os.getenv("DEV_DB_PASSWORD")
    if pw:
        return pw
    pw_file = os.getenv("DEV_DB_PASSWORD_FILE")
    if pw_file:
        return open(pw_file).read().strip()
    raise RuntimeError("No DB password provided")
