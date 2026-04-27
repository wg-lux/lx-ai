from django.db import connection, transaction
import uuid

SOURCE_VIDEO_ID = 1

# First run with True.
# After reviewing output, change to False and run again.
DRY_RUN = False


def make_video_hash(old_exam_id: int) -> str:
    return f"split_exam_{old_exam_id}_{uuid.uuid4().hex}"


def get_table_columns(table_name: str) -> list[str]:
    if connection.vendor == "sqlite":
        with connection.cursor() as cursor:
            cursor.execute(f"PRAGMA table_info({table_name})")
            return [row[1] for row in cursor.fetchall()]

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
            """,
            [table_name],
        )
        return [row[0] for row in cursor.fetchall()]


def insert_videofile_and_return_id(insert_sql: str, insert_vals: list[object]) -> int:
    with connection.cursor() as cursor:
        if connection.vendor == "postgresql":
            cursor.execute(insert_sql + " RETURNING id", insert_vals)
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("INSERT succeeded but no id was returned.")
            new_video_id = row[0]
        else:
            cursor.execute(insert_sql, insert_vals)
            new_video_id = cursor.lastrowid

    if new_video_id is None:
        raise RuntimeError("Failed to obtain new videofile id after INSERT.")

    return int(new_video_id)


def fetch_video_columns() -> list[str]:
    return get_table_columns("endoreg_db_videofile")


def fetch_source_video_row(source_video_id: int, copy_cols: list[str]):
    sql = f"""
        SELECT {", ".join(copy_cols)}
        FROM endoreg_db_videofile
        WHERE id = %s
    """
    with connection.cursor() as cursor:
        cursor.execute(sql, [source_video_id])
        return cursor.fetchone()


def fetch_exam_groups(source_video_id: int):
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT old_examination_id, COUNT(*) AS num_frames
            FROM endoreg_db_frame
            WHERE video_id = %s
              AND old_examination_id IS NOT NULL
            GROUP BY old_examination_id
            ORDER BY old_examination_id
        """, [source_video_id])
        return cursor.fetchall()


def fetch_null_exam_count(source_video_id: int):
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*)
            FROM endoreg_db_frame
            WHERE video_id = %s
              AND old_examination_id IS NULL
        """, [source_video_id])
        return cursor.fetchone()[0]


def fetch_total_frame_count(source_video_id: int):
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*)
            FROM endoreg_db_frame
            WHERE video_id = %s
        """, [source_video_id])
        return cursor.fetchone()[0]


def fetch_video1_summary(source_video_id: int):
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(DISTINCT old_examination_id)
            FROM endoreg_db_frame
            WHERE video_id = %s
              AND old_examination_id IS NOT NULL
        """, [source_video_id])
        unique_group_count = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM endoreg_db_frame
            WHERE video_id = %s
              AND old_examination_id IS NOT NULL
        """, [source_video_id])
        frames_with_old_exam = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM endoreg_db_frame
            WHERE video_id = %s
              AND old_examination_id IS NULL
        """, [source_video_id])
        frames_without_old_exam = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM endoreg_db_frame
            WHERE video_id = %s
        """, [source_video_id])
        total_frames = cursor.fetchone()[0]

    return {
        "unique_group_count": unique_group_count,
        "frames_with_old_exam": frames_with_old_exam,
        "frames_without_old_exam": frames_without_old_exam,
        "total_frames": total_frames,
    }


def validate_after_split(source_video_id: int):
    print("\n=== POST-RUN VALIDATION ===")
    print("-" * 80)

    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*)
            FROM endoreg_db_frame
        """)
        total_frames_global = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM endoreg_db_frame
            WHERE video_id = %s
        """, [source_video_id])
        remaining_on_source = cursor.fetchone()[0]

        cursor.execute("""
            SELECT video_id, COUNT(DISTINCT old_examination_id) AS num_exam_groups, COUNT(*) AS num_frames
            FROM endoreg_db_frame
            GROUP BY video_id
            ORDER BY video_id
        """)
        per_video_groups = cursor.fetchall()

    print(f"Global total frames               : {total_frames_global}")
    print(f"Frames still on source video {source_video_id}: {remaining_on_source}")
    print("\nvideo_id | distinct old_examination_id groups | frame_count")
    for row in per_video_groups[:50]:
        print(row)
    if len(per_video_groups) > 50:
        print(f"... ({len(per_video_groups) - 50} more rows)")


def run():
    print("\n" + "=" * 100)
    print("SPLIT source video by old_examination_id")
    print("=" * 100)

    video_cols = fetch_video_columns()

    required_cols = set(video_cols)

    # Copy almost everything from source video row except uniqueness-sensitive columns.
    force_null_if_present = {
        "processed_video_hash",
        "sensitive_meta_id",
        "import_meta_id",
        "video_meta_id",
        "state_id",
    }

    override_cols = {
        "uuid",
        "video_hash",
    }

    copy_cols = []
    for col in video_cols:
        if col == "id":
            continue
        if col in force_null_if_present:
            continue
        if col in override_cols:
            continue
        copy_cols.append(col)

    source_row = fetch_source_video_row(SOURCE_VIDEO_ID, copy_cols)
    if source_row is None:
        raise ValueError(f"Source videofile id={SOURCE_VIDEO_ID} not found")

    summary = fetch_video1_summary(SOURCE_VIDEO_ID)
    exam_groups = fetch_exam_groups(SOURCE_VIDEO_ID)
    null_exam_count = fetch_null_exam_count(SOURCE_VIDEO_ID)

    print("\n=== PRE-RUN SUMMARY ===")
    print("-" * 80)
    print(f"Source video_id                    : {SOURCE_VIDEO_ID}")
    print(f"Unique old_examination_id groups   : {summary['unique_group_count']}")
    print(f"Frames with old_examination_id     : {summary['frames_with_old_exam']}")
    print(f"Frames with NULL old_examination_id: {summary['frames_without_old_exam']}")
    print(f"Total frames on source video       : {summary['total_frames']}")
    print(f"New videofile rows to create       : {len(exam_groups)}")
    print(f"Frames staying on original video   : {null_exam_count}")

    print("\nTop 20 groups by size:")
    top20 = sorted(exam_groups, key=lambda x: (-x[1], x[0]))[:20]
    for old_exam_id, num_frames in top20:
        print(f"old_examination_id={old_exam_id} | frames={num_frames}")

    if DRY_RUN:
        print("\nDRY RUN ONLY — no DB changes made.")
        print("=" * 100)
        return

    created_count = 0
    moved_count = 0
    mapping = []

    with transaction.atomic():
        for old_exam_id, num_frames in exam_groups:
            new_uuid = uuid.uuid4().hex
            new_video_hash = make_video_hash(old_exam_id)

            insert_cols = list(copy_cols)
            insert_vals = list(source_row)

            for col in sorted(force_null_if_present):
                if col in required_cols:
                    insert_cols.append(col)
                    insert_vals.append(None)

            if "uuid" in required_cols:
                insert_cols.append("uuid")
                insert_vals.append(new_uuid)

            if "video_hash" in required_cols:
                insert_cols.append("video_hash")
                insert_vals.append(new_video_hash)

            placeholders = ", ".join(["%s"] * len(insert_cols))
            insert_sql = f"""
                INSERT INTO endoreg_db_videofile ({", ".join(insert_cols)})
                VALUES ({placeholders})
            """

            new_video_id = insert_videofile_and_return_id(
                insert_sql=insert_sql,
                insert_vals=insert_vals,
            )

            with connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE endoreg_db_frame
                    SET video_id = %s
                    WHERE video_id = %s
                      AND old_examination_id = %s
                """, [new_video_id, SOURCE_VIDEO_ID, old_exam_id])

            created_count += 1
            moved_count += num_frames
            mapping.append((old_exam_id, new_video_id, num_frames))

            print(
                f"Created new video_id={new_video_id} "
                f"for old_examination_id={old_exam_id} "
                f"| moved_frames={num_frames}"
            )

    print("\n=== EXECUTION SUMMARY ===")
    print("-" * 80)
    print(f"New videofile rows created : {created_count}")
    print(f"Frames moved               : {moved_count}")
    print(f"Frames left on source video: {null_exam_count}")

    print("\nSample mapping (first 30):")
    for row in mapping[:30]:
        print(row)
    if len(mapping) > 30:
        print(f"... ({len(mapping) - 30} more rows)")

    validate_after_split(SOURCE_VIDEO_ID)

    print("\nDONE.")
    print("=" * 100)


if __name__ == "__main__":
    run()