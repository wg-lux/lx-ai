# Dry-run queries for aidataset_id = 1

Purpose: inspect counts and relationships before exporting. Run these in your `psql` session and paste results.

Notes:
- DO NOT EXPORT YET — this is a dry run.
- Run all queries and share results before proceeding to COPY/dump steps.

---

## 1) Check dataset exists

```sql
SELECT *
FROM endoreg_db_aidataset
WHERE id = 1;
```

---

## 2) Count linked annotations

```sql
SELECT COUNT(*) AS annotation_count
FROM endoreg_db_aidataset_image_annotations dai
JOIN endoreg_db_imageclassificationannotation a
    ON a.id = dai.imageclassificationannotation_id
WHERE dai.aidataset_id = 1;
```

---

## 3) Count unique frames

```sql
SELECT COUNT(DISTINCT a.frame_id) AS frame_count
FROM endoreg_db_aidataset_image_annotations dai
JOIN endoreg_db_imageclassificationannotation a
    ON a.id = dai.imageclassificationannotation_id
WHERE dai.aidataset_id = 1;
```

---

## 4) Count unique videofiles (videos)

```sql
SELECT COUNT(DISTINCT f.video_id) AS video_count
FROM endoreg_db_aidataset_image_annotations dai
JOIN endoreg_db_imageclassificationannotation a
    ON a.id = dai.imageclassificationannotation_id
JOIN endoreg_db_frame f
    ON f.id = a.frame_id
WHERE dai.aidataset_id = 1;
```

---

## 5) Count used labels

```sql
SELECT COUNT(DISTINCT a.label_id) AS label_count
FROM endoreg_db_aidataset_image_annotations dai
JOIN endoreg_db_imageclassificationannotation a
    ON a.id = dai.imageclassificationannotation_id
WHERE dai.aidataset_id = 1;
```

---

## 6) Inspect which labels

```sql
SELECT DISTINCT l.id, l.name
FROM endoreg_db_aidataset_image_annotations dai
JOIN endoreg_db_imageclassificationannotation a
    ON a.id = dai.imageclassificationannotation_id
JOIN endoreg_db_label l
    ON l.id = a.label_id
WHERE dai.aidataset_id = 1
ORDER BY l.id;
```

---

## 7) Identify labelset(s) used

```sql
SELECT DISTINCT ls.id, ls.version
FROM endoreg_db_label l
JOIN endoreg_db_labelset_labels lsl
    ON lsl.label_id = l.id
JOIN endoreg_db_labelset ls
    ON ls.id = lsl.labelset_id
WHERE l.id IN (
    SELECT DISTINCT a.label_id
    FROM endoreg_db_aidataset_image_annotations dai
    JOIN endoreg_db_imageclassificationannotation a
        ON a.id = dai.imageclassificationannotation_id
    WHERE dai.aidataset_id = 1
);
```

---

Run these and paste the results. I will then provide the filtered COPY/dump commands in dependency-safe order for exporting only data related to `aidataset_id = 1`.