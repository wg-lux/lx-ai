\set ON_ERROR_STOP on

BEGIN;

-- =========================================================
-- 0) CLEAN STAGING
-- =========================================================

DROP SCHEMA IF EXISTS stg_import CASCADE;
CREATE SCHEMA stg_import;

CREATE TABLE stg_import.videofile (
    id bigint,
    raw_file text,
    processed_file text,
    video_hash text,
    processed_video_hash text,
    original_file_name text,
    uploaded_at timestamptz,
    frame_dir text,
    fps double precision,
    duration double precision,
    frame_count integer,
    width integer,
    height integer,
    suffix text,
    sequences jsonb,
    date date,
    meta jsonb,
    date_created timestamptz,
    date_modified timestamptz,
    ai_model_meta_id bigint,
    center_id bigint,
    examination_id bigint,
    patient_id bigint,
    processor_id bigint,
    sensitive_meta_id bigint,
    import_meta_id bigint,
    video_meta_id bigint,
    state_id bigint,
    export_segments_by_video boolean,
    uuid uuid
);

CREATE TABLE stg_import.frame (
    id bigint,
    frame_number integer,
    relative_path text,
    timestamp double precision,
    is_extracted boolean,
    video_id bigint
);

CREATE TABLE stg_import.annotation (
    id bigint,
    value boolean,
    float_value double precision,
    annotator text,
    date_created timestamptz,
    date_modified timestamptz,
    frame_id bigint,
    information_source_id bigint,
    label_id bigint,
    model_meta_id bigint,
    external_annotation_id bigint
);

\copy stg_import.annotation FROM '/home/admin/csv/01_annotation.csv' WITH CSV HEADER
\copy stg_import.frame      FROM '/home/admin/csv/02_frame.csv'      WITH CSV HEADER
\copy stg_import.videofile  FROM '/home/admin/csv/03_videofile.csv'  WITH CSV HEADER

-- =========================================================
-- 1) STAGING VALIDATION
-- =========================================================

-- annotation.frame_id must exist in imported frames
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM stg_import.annotation a
        LEFT JOIN stg_import.frame f ON f.id = a.frame_id
        WHERE f.id IS NULL
    ) THEN
        RAISE EXCEPTION 'Some annotation.frame_id values do not exist in stg_import.frame';
    END IF;
END $$;

-- frame.video_id must exist in imported videos
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM stg_import.frame f
        LEFT JOIN stg_import.videofile v ON v.id = f.video_id
        WHERE v.id IS NULL
    ) THEN
        RAISE EXCEPTION 'Some frame.video_id values do not exist in stg_import.videofile';
    END IF;
END $$;

-- label_id must already exist in target
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM stg_import.annotation a
        LEFT JOIN endoreg_db_label l ON l.id = a.label_id
        WHERE l.id IS NULL
    ) THEN
        RAISE EXCEPTION 'Some annotation.label_id values do not exist in target endoreg_db_label';
    END IF;
END $$;

-- center_id must exist in target when non-null
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM stg_import.videofile v
        LEFT JOIN endoreg_db_center c ON c.id = v.center_id
        WHERE v.center_id IS NOT NULL
          AND c.id IS NULL
    ) THEN
        RAISE EXCEPTION 'Some videofile.center_id values do not exist in target endoreg_db_center';
    END IF;
END $$;

-- processor_id must exist in target when non-null
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM stg_import.videofile v
        LEFT JOIN endoreg_db_endoscopyprocessor p ON p.id = v.processor_id
        WHERE v.processor_id IS NOT NULL
          AND p.id IS NULL
    ) THEN
        RAISE EXCEPTION 'Some videofile.processor_id values do not exist in target endoreg_db_endoscopyprocessor';
    END IF;
END $$;

-- information_source_id must exist in target when non-null
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM stg_import.annotation a
        LEFT JOIN endoreg_db_informationsource s ON s.id = a.information_source_id
        WHERE a.information_source_id IS NOT NULL
          AND s.id IS NULL
    ) THEN
        RAISE EXCEPTION 'Some annotation.information_source_id values do not exist in target endoreg_db_informationsource';
    END IF;
END $$;

-- uuid must not already exist in target
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM stg_import.videofile s
        JOIN endoreg_db_videofile t ON t.uuid = s.uuid
    ) THEN
        RAISE EXCEPTION 'Some videofile.uuid values already exist in target endoreg_db_videofile';
    END IF;
END $$;

-- video_hash must not already exist in target
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM stg_import.videofile s
        JOIN endoreg_db_videofile t ON t.video_hash = s.video_hash
    ) THEN
        RAISE EXCEPTION 'Some videofile.video_hash values already exist in target endoreg_db_videofile';
    END IF;
END $$;

-- processed_video_hash must not already exist in target when non-null/non-empty
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM stg_import.videofile s
        JOIN endoreg_db_videofile t
          ON t.processed_video_hash = s.processed_video_hash
        WHERE s.processed_video_hash IS NOT NULL
          AND s.processed_video_hash <> ''
    ) THEN
        RAISE EXCEPTION 'Some videofile.processed_video_hash values already exist in target endoreg_db_videofile';
    END IF;
END $$;

-- optional: reject duplicate uuids inside CSV itself
DO $$
BEGIN
    IF EXISTS (
        SELECT uuid
        FROM stg_import.videofile
        GROUP BY uuid
        HAVING COUNT(*) > 1
    ) THEN
        RAISE EXCEPTION 'Duplicate uuid values found inside stg_import.videofile';
    END IF;
END $$;

-- optional: reject duplicate video_hash inside CSV itself
DO $$
BEGIN
    IF EXISTS (
        SELECT video_hash
        FROM stg_import.videofile
        GROUP BY video_hash
        HAVING COUNT(*) > 1
    ) THEN
        RAISE EXCEPTION 'Duplicate video_hash values found inside stg_import.videofile';
    END IF;
END $$;

-- =========================================================
-- 2) CREATE NEW DATASET
-- =========================================================

CREATE TEMP TABLE tmp_new_dataset (id bigint);

WITH ins AS (
    INSERT INTO endoreg_db_aidataset (
        name,
        description,
        ai_model_type,
        dataset_type,
        created_at,
        updated_at,
        is_active
    )
    VALUES (
        'csv_import_' || to_char(NOW(), 'YYYYMMDD_HH24MISS'),
        'Imported from CSV files in /home/admin/csv',
        'image_multilabel_classification',
        'image',
        NOW(),
        NOW(),
        TRUE
    )
    RETURNING id
)
INSERT INTO tmp_new_dataset
SELECT id FROM ins;

-- =========================================================
-- 3) CREATE MAPPING TABLES
-- =========================================================

CREATE TEMP TABLE map_videofile (
    old_id bigint PRIMARY KEY,
    new_id bigint NOT NULL
);

CREATE TEMP TABLE map_frame (
    old_id bigint PRIMARY KEY,
    new_id bigint NOT NULL
);

CREATE TEMP TABLE map_annotation (
    old_id bigint PRIMARY KEY,
    new_id bigint NOT NULL
);

-- =========================================================
-- 4) INSERT VIDEOFILES WITH PRESERVED UUID/HASH
-- =========================================================

WITH src AS (
    SELECT
        v.*,
        row_number() OVER (ORDER BY v.id) AS rn
    FROM stg_import.videofile v
),
ins AS (
    INSERT INTO endoreg_db_videofile (
        raw_file,
        processed_file,
        uuid,
        video_hash,
        processed_video_hash,
        sensitive_meta_id,
        center_id,
        processor_id,
        video_meta_id,
        examination_id,
        patient_id,
        ai_model_meta_id,
        state_id,
        import_meta_id,
        original_file_name,
        uploaded_at,
        frame_dir,
        fps,
        duration,
        frame_count,
        width,
        height,
        suffix,
        sequences,
        export_segments_by_video,
        date,
        meta,
        date_created,
        date_modified
    )
    SELECT
        NULLIF(src.raw_file, ''),
        NULLIF(src.processed_file, ''),
        src.uuid,
        src.video_hash,
        NULLIF(src.processed_video_hash, ''),
        NULL,  -- sensitive_meta_id
        src.center_id,
        CASE
            WHEN src.processor_id IS NOT NULL
             AND EXISTS (
                 SELECT 1
                 FROM endoreg_db_endoscopyprocessor p
                 WHERE p.id = src.processor_id
             )
            THEN src.processor_id
            ELSE NULL
        END,
        NULL,  -- video_meta_id
        NULL,  -- examination_id
        NULL,  -- patient_id
        NULL,  -- ai_model_meta_id
        NULL,  -- state_id
        NULL,  -- import_meta_id
        src.original_file_name,
        COALESCE(src.uploaded_at, NOW()),
        src.frame_dir,
        src.fps,
        src.duration,
        src.frame_count,
        src.width,
        src.height,
        src.suffix,
        COALESCE(src.sequences, '{}'::jsonb),
        COALESCE(src.export_segments_by_video, FALSE),
        src.date,
        src.meta,
        COALESCE(src.date_created, NOW()),
        COALESCE(src.date_modified, NOW())
    FROM src
    ORDER BY src.rn
    RETURNING id
),
ins_numbered AS (
    SELECT
        id,
        row_number() OVER (ORDER BY id) AS rn
    FROM ins
)
INSERT INTO map_videofile (old_id, new_id)
SELECT
    src.id,
    ins_numbered.id
FROM src
JOIN ins_numbered USING (rn);

-- =========================================================
-- 5) INSERT FRAMES WITH REMAPPED VIDEO_ID
-- =========================================================

WITH src AS (
    SELECT
        f.*,
        row_number() OVER (ORDER BY f.id) AS rn
    FROM stg_import.frame f
),
ins AS (
    INSERT INTO endoreg_db_frame (
        video_id,
        frame_number,
        relative_path,
        timestamp,
        is_extracted
    )
    SELECT
        mv.new_id,
        src.frame_number,
        src.relative_path,
        src.timestamp,
        COALESCE(src.is_extracted, FALSE)
    FROM src
    JOIN map_videofile mv ON mv.old_id = src.video_id
    ORDER BY src.rn
    RETURNING id
),
ins_numbered AS (
    SELECT
        id,
        row_number() OVER (ORDER BY id) AS rn
    FROM ins
)
INSERT INTO map_frame (old_id, new_id)
SELECT
    src.id,
    ins_numbered.id
FROM src
JOIN ins_numbered USING (rn);

-- =========================================================
-- 6) INSERT ANNOTATIONS WITH REMAPPED FRAME_ID
-- =========================================================

WITH src AS (
    SELECT
        a.*,
        row_number() OVER (ORDER BY a.id) AS rn
    FROM stg_import.annotation a
),
ins AS (
    INSERT INTO endoreg_db_imageclassificationannotation (
        frame_id,
        label_id,
        value,
        float_value,
        annotator,
        model_meta_id,
        date_created,
        date_modified,
        information_source_id
    )
    SELECT
        mf.new_id,
        src.label_id,
        src.value,
        src.float_value,
        src.annotator,
        NULL,  -- model_meta_id
        COALESCE(src.date_created, NOW()),
        COALESCE(src.date_modified, NOW()),
        CASE
            WHEN src.information_source_id IS NOT NULL
             AND EXISTS (
                 SELECT 1
                 FROM endoreg_db_informationsource s
                 WHERE s.id = src.information_source_id
             )
            THEN src.information_source_id
            ELSE NULL
        END
    FROM src
    JOIN map_frame mf ON mf.old_id = src.frame_id
    ORDER BY src.rn
    RETURNING id
),
ins_numbered AS (
    SELECT
        id,
        row_number() OVER (ORDER BY id) AS rn
    FROM ins
)
INSERT INTO map_annotation (old_id, new_id)
SELECT
    src.id,
    ins_numbered.id
FROM src
JOIN ins_numbered USING (rn);

-- =========================================================
-- 7) INSERT PIVOT ROWS FOR NEW DATASET
-- =========================================================

INSERT INTO endoreg_db_aidataset_image_annotations (
    aidataset_id,
    imageclassificationannotation_id
)
SELECT
    (SELECT id FROM tmp_new_dataset LIMIT 1),
    ma.new_id
FROM map_annotation ma
ORDER BY ma.new_id;

-- =========================================================
-- 8) VALIDATION
-- =========================================================

SELECT 'new_dataset_id' AS info, id
FROM tmp_new_dataset;

SELECT 'staging_videofile_count' AS info, COUNT(*)::bigint FROM stg_import.videofile
UNION ALL
SELECT 'staging_frame_count', COUNT(*)::bigint FROM stg_import.frame
UNION ALL
SELECT 'staging_annotation_count', COUNT(*)::bigint FROM stg_import.annotation;

SELECT 'imported_pivot_count' AS info, COUNT(*)::bigint
FROM endoreg_db_aidataset_image_annotations
WHERE aidataset_id = (SELECT id FROM tmp_new_dataset LIMIT 1)
UNION ALL
SELECT 'imported_annotation_distinct', COUNT(DISTINCT a.id)::bigint
FROM endoreg_db_aidataset_image_annotations dai
JOIN endoreg_db_imageclassificationannotation a
  ON a.id = dai.imageclassificationannotation_id
WHERE dai.aidataset_id = (SELECT id FROM tmp_new_dataset LIMIT 1)
UNION ALL
SELECT 'imported_frame_distinct', COUNT(DISTINCT f.id)::bigint
FROM endoreg_db_aidataset_image_annotations dai
JOIN endoreg_db_imageclassificationannotation a
  ON a.id = dai.imageclassificationannotation_id
JOIN endoreg_db_frame f
  ON f.id = a.frame_id
WHERE dai.aidataset_id = (SELECT id FROM tmp_new_dataset LIMIT 1)
UNION ALL
SELECT 'imported_videofile_distinct', COUNT(DISTINCT v.id)::bigint
FROM endoreg_db_aidataset_image_annotations dai
JOIN endoreg_db_imageclassificationannotation a
  ON a.id = dai.imageclassificationannotation_id
JOIN endoreg_db_frame f
  ON f.id = a.frame_id
JOIN endoreg_db_videofile v
  ON v.id = f.video_id
WHERE dai.aidataset_id = (SELECT id FROM tmp_new_dataset LIMIT 1);

SELECT 'map_videofile_count' AS info, COUNT(*)::bigint FROM map_videofile
UNION ALL
SELECT 'map_frame_count', COUNT(*)::bigint FROM map_frame
UNION ALL
SELECT 'map_annotation_count', COUNT(*)::bigint FROM map_annotation;

-- =========================================================
-- 9) DRY RUN OR COMMIT
-- =========================================================

-- COMMIT;
ROLLBACK;