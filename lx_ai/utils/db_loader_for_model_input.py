# lx_ai/utils/db_loader_for_model_input.py
from __future__ import annotations

import os
from typing import Optional

import psycopg


def _first_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    """
    Return the first non-empty environment variable value from names.
    Empty strings are treated as missing.
    """
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default


'''def _get_password() -> str:
    """
    Password resolution order:
    1. DEV_DB_PASSWORD
    2. DJANGO_DB_PASSWORD
    3. DEV_DB_PASSWORD_FILE
    4. DJANGO_DB_PASSWORD_FILE

    This keeps local override support while remaining compatible with service.
    """
    pw = _first_env("DEV_DB_PASSWORD", "DJANGO_DB_PASSWORD")
    if pw is not None:
        return pw

    pw_file = _first_env("DEV_DB_PASSWORD_FILE", "DJANGO_DB_PASSWORD_FILE")
    if pw_file is not None:
        with open(pw_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    raise RuntimeError(
        "No database password provided. Set one of: "
        "DEV_DB_PASSWORD, DJANGO_DB_PASSWORD, "
        "DEV_DB_PASSWORD_FILE, or DJANGO_DB_PASSWORD_FILE."
    )'''
def _get_password() -> str:
    """
    Resolve DB password safely for both local and service.

    Order:
    1. DEV_DB_PASSWORD / DJANGO_DB_PASSWORD
    2. DEV_DB_PASSWORD_FILE / DJANGO_DB_PASSWORD_FILE (if file exists)
    """

    # 1️⃣ direct password (best for local)
    pw = _first_env("DEV_DB_PASSWORD", "DJANGO_DB_PASSWORD")
    if pw:
        return pw

    # 2️⃣ password file (used by service)
    pw_file = _first_env("DEV_DB_PASSWORD_FILE", "DJANGO_DB_PASSWORD_FILE")
    if pw_file:
        if os.path.exists(pw_file):
            with open(pw_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        else:
            raise RuntimeError(
                f"Password file '{pw_file}' not found.\n"
                "For local dev: set DEV_DB_PASSWORD.\n"
                "For service: ensure secretspec creates the file."
            )

    # 3️⃣ nothing found
    raise RuntimeError(
        "No database password found.\n"
        "Set one of:\n"
        "- DEV_DB_PASSWORD (local)\n"
        "- DJANGO_DB_PASSWORD\n"
        "- DEV_DB_PASSWORD_FILE\n"
        "- DJANGO_DB_PASSWORD_FILE"
    )

def _get_db_connection_kwargs() -> dict:
    """
    Resolve connection settings with local-first, service-compatible fallback.

    Precedence:
      DEV_DB_*    -> local explicit override
      DJANGO_DB_* -> service/runtime compatibility
      defaults    -> safe local fallback
    """
    host = _first_env("DEV_DB_HOST", "DJANGO_DB_HOST", default="localhost")
    port_str = _first_env("DEV_DB_PORT", "DJANGO_DB_PORT", default="5432")
    dbname = _first_env("DEV_DB_NAME", "DJANGO_DB_NAME", default="endoregDbLocal")
    user = _first_env("DEV_DB_USER", "DJANGO_DB_USER", default="endoregDbLocal")
    sslmode = _first_env("DEV_DB_SSLMODE", "DJANGO_DB_SSLMODE", default="disable")

    try:
        port = int(port_str)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid database port: {port_str!r}. "
            "Set DEV_DB_PORT or DJANGO_DB_PORT to an integer."
        ) from exc

    password = _get_password()

    return {
        "host": host,
        "port": port,
        "dbname": dbname,
        "user": user,
        "password": password,
        "sslmode": sslmode,
    }


def load_annotations_from_postgres(dataset_id: int) -> list[dict]:
    sql = """
    SELECT
        dai.aidataset_id        AS aidataset_id,
        f.id                    AS frame_id,
        f.relative_path         AS relative_path,
        vf.frame_dir            AS frame_dir,
        f.old_examination_id    AS old_examination_id,
        vf.id                   AS video_id,
        l.id                    AS label_id,
        l.name                  AS label_name,
        a.value                 AS value,
        a.annotator             AS annotator
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
    conn_kwargs = _get_db_connection_kwargs()

    with psycopg.connect(**conn_kwargs) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (dataset_id,))
            for row in cur.fetchall():
                rows.append(
                    {
                        "dataset_id": row[0],
                        "frame": {
                            "id": row[1],
                            "relative_path": row[2],
                            "file_path": row[3],
                            "old_examination_id": row[4],
                            "video_id": row[5],
                        },
                        "label": {
                            "id": row[6],
                            "name": row[7],
                        },
                        "value": row[8],
                        "annotator": row[9],
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
    conn_kwargs = _get_db_connection_kwargs()

    with psycopg.connect(**conn_kwargs) as conn:
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


import os

def load_annotations(config, dataset_id: int) -> list[dict]:
    db_backend = os.getenv("DB_BACKEND", "postgres")

    if db_backend == "sqlite":
        return load_annotations_from_sqlite(dataset_id)
    elif db_backend == "postgres":
        return load_annotations_from_postgres(dataset_id)
    else:
        raise ValueError(f"Unsupported DB backend: {db_backend}")
    
import sqlite3
from pathlib import Path


def load_annotations_from_sqlite(dataset_id: int) -> list[dict]:
    db_path = Path(os.getenv("SQLITE_DB_PATH", "dev_db.sqlite")).expanduser()

    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    sql = """
    SELECT
        dai.aidataset_id,
        f.id,
        f.relative_path,
        vf.frame_dir,
        f.old_examination_id,
        vf.id,
        l.id,
        l.name,
        a.value,
        a.annotator
    FROM endoreg_db_aidataset_image_annotations dai
    JOIN endoreg_db_imageclassificationannotation a
        ON a.id = dai.imageclassificationannotation_id
    JOIN endoreg_db_frame f
        ON f.id = a.frame_id
    JOIN endoreg_db_videofile vf
        ON vf.id = f.video_id
    JOIN endoreg_db_label l
        ON l.id = a.label_id
    WHERE dai.aidataset_id = ?
    """

    rows: list[dict] = []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(sql, (dataset_id,))

    for row in cursor.fetchall():
        rows.append(
            {
                "dataset_id": row[0],
                "frame": {
                    "id": row[1],
                    "relative_path": row[2],
                    "file_path": row[3],
                    "old_examination_id": row[4],
                    "video_id": row[5],
                },
                "label": {
                    "id": row[6],
                    "name": row[7],
                },
                "value": row[8],
                "annotator": row[9],
            }
        )

    conn.close()
    return rows