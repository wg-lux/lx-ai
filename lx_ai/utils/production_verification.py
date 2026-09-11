from __future__ import annotations

import argparse
import importlib
import inspect
import os
from pathlib import Path
from typing import Any


FAILURES: list[str] = []
WARNINGS: list[str] = []


def section(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def ok(message: str) -> None:
    print(f"OK   : {message}")


def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"WARN : {message}")


def fail(message: str) -> None:
    FAILURES.append(message)
    print(f"FAIL : {message}")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def load_env_file(path: Path) -> None:
    if not path.exists():
        fail(f"Environment file not found: {path}")
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = raw_line.split("=", 1)
        key = key.strip()

        if not key.replace("_", "").isalnum() or key[0].isdigit():
            warn(f"Skipping invalid env key: {key}")
            continue

        os.environ[key] = value


def load_db_password_from_file() -> None:
    if env("DJANGO_DB_PASSWORD"):
        return

    for key in ("DJANGO_DB_PASSWORD_FILE", "DB_PWD_FILE"):
        raw = env(key)
        if not raw:
            continue

        path = Path(raw)
        if path.exists():
            os.environ["DJANGO_DB_PASSWORD"] = path.read_text(encoding="utf-8").strip()
            ok(f"DJANGO_DB_PASSWORD loaded from {key}, value not printed")
            return

    warn("DJANGO_DB_PASSWORD not loaded because no readable password file was found")


def check_path(
    name: str,
    *,
    required: bool = True,
    directory: bool | None = None,
    readable: bool = False,
    writable: bool = False,
) -> Path | None:
    raw = env(name)

    if not raw:
        message = f"{name} is not set"
        if required:
            fail(message)
        else:
            warn(message)
        return None

    path = Path(raw).expanduser()
    exists = path.exists()

    print(f"{name:<40} = {raw} | exists={exists}")

    if required and not exists:
        fail(f"{name} does not exist: {path}")
        return path

    if exists and directory is True and not path.is_dir():
        fail(f"{name} must be a directory: {path}")

    if exists and directory is False and not path.is_file():
        fail(f"{name} must be a file: {path}")

    if exists and readable and not os.access(path, os.R_OK):
        fail(f"{name} is not readable: {path}")

    if exists and writable:
        target_dir = path if path.is_dir() else path.parent
        probe = target_dir / ".lxai_verify_write_test"

        try:
            probe.write_text("ok\n", encoding="utf-8")
            probe.unlink(missing_ok=True)
            ok(f"{name} is writable")
        except Exception as exc:
            fail(f"{name} write check failed: {type(exc).__name__}: {exc}")

    return path


def import_module(name: str, *, required: bool = True) -> Any | None:
    try:
        module = importlib.import_module(name)
        ok(f"import {name}")
        return module
    except Exception as exc:
        message = f"import {name} failed: {type(exc).__name__}: {exc}"
        if required:
            fail(message)
        else:
            warn(message)
        return None


def get_model_by_name(model_name: str):
    from django.apps import apps

    for model in apps.get_models():
        if model.__name__ == model_name:
            return model

    return None


def model_count(model_name: str) -> int | None:
    model = get_model_by_name(model_name)

    if model is None:
        fail(f"Django model not found: {model_name}")
        return None

    try:
        return int(model.objects.count())
    except Exception as exc:
        fail(f"Could not count {model_name}: {type(exc).__name__}: {exc}")
        return None


def find_key_recursive(obj: Any, keys: set[str]) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in keys:
                return value

        for value in obj.values():
            found = find_key_recursive(value, keys)
            if found is not None:
                return found

    if isinstance(obj, list):
        for value in obj:
            found = find_key_recursive(value, keys)
            if found is not None:
                return found

    return None


def verify_paths() -> None:
    section("1. SERVICE ENVIRONMENT AND PATHS")

    for name in [
        "HOME_DIR",
        "WORKING_DIR",
        "DATA_DIR",
        "DJANGO_DATA_DIR",
        "LX_ANNOTATE_DATA_DIR",
        "LX_ANNOTATE_ENCRYPTED_DATA_DIR",
        "STORAGE_DIR",
        "PROTECTED_MEDIA_ROOT",
        "CONF_DIR",
        "FRAME_DIR",
        "FRAME_MATERIALIZATION_OUTPUT_ROOT",
        "TRAINING_ROOT",
        "CHECKPOINTS_DIR",
        "RUNS_DIR",
        "BUCKET_SNAPSHOT_DIR",
        "ENDOREG_DB_PLAINTEXT_TMP_DIR",
    ]:
        check_path(
            name,
            required=True,
            directory=True,
            readable=name
            in {
                "LX_ANNOTATE_ENCRYPTED_DATA_DIR",
                "STORAGE_DIR",
                "PROTECTED_MEDIA_ROOT",
            },
            writable=name
            in {
                "DATA_DIR",
                "FRAME_DIR",
                "FRAME_MATERIALIZATION_OUTPUT_ROOT",
                "TRAINING_ROOT",
                "CHECKPOINTS_DIR",
                "RUNS_DIR",
                "BUCKET_SNAPSHOT_DIR",
                "ENDOREG_DB_PLAINTEXT_TMP_DIR",
            },
        )

    print()
    for name in [
        "DJANGO_ENV",
        "DJANGO_SETTINGS_MODULE",
        "DB_BACKEND",
        "DJANGO_DB_ENGINE",
        "DJANGO_DB_NAME",
        "DJANGO_DB_USER",
        "DJANGO_DB_HOST",
        "DJANGO_DB_PORT",
        "DJANGO_DB_SSLMODE",
        "TRAINING_CONFIG_PATH",
    ]:
        print(f"{name:<40} = {env(name, '<unset>')}")

    check_path("DJANGO_DB_PASSWORD_FILE", required=True, directory=False, readable=True)
    check_path(
        "LX_ANNOTATE_MASTER_KEY_FILE", required=True, directory=False, readable=True
    )


def verify_imports_before_django() -> None:
    section("2. SAFE PYTHON IMPORTS BEFORE DJANGO SETUP")

    for module_name in [
        "django",
        "lx_ai",
        "endoreg_db",
        "lx_dtypes",
        "yaml",
    ]:
        import_module(module_name)


def setup_django() -> None:
    section("3. DJANGO SETUP AND DATABASE")

    try:
        import django

        django.setup()
        ok("django.setup() completed")
    except Exception as exc:
        fail(f"django.setup() failed: {type(exc).__name__}: {exc}")
        return

    try:
        from django.db import connection

        with connection.cursor() as cursor:
            cursor.execute("select current_database(), current_user")
            database_name, database_user = cursor.fetchone()

        ok(
            f"PostgreSQL connection works: database={database_name}, user={database_user}"
        )
        ok(f"Django vendor={connection.vendor}")

        if (
            env("DJANGO_ENV").lower() == "production"
            and connection.vendor != "postgresql"
        ):
            fail(f"Production must use PostgreSQL, got {connection.vendor}")

    except Exception as exc:
        fail(f"Database connection failed: {type(exc).__name__}: {exc}")


def verify_imports_after_django() -> None:
    section("4. DJANGO-DEPENDENT IMPORTS AFTER DJANGO SETUP")

    for module_name in [
        "lx_ai.utils.frame_materializer",
        "lx_ai.utils.endoregdb_encrypted_frame_bridge",
        "endoreg_db.utils.paths",
        "endoreg_db.export.frames.export_frames_with_labels",
        "lx_ai.utils.db_loader_for_model_input",
        "lx_ai.utils.data_loader_for_model_input",
    ]:
        import_module(module_name)


def verify_endoreg_paths() -> None:
    section("5. ENDOREG-DB PATH CONTRACT")

    try:
        from endoreg_db.config import env as endoreg_env
        from endoreg_db.utils.paths import (
            EndoregPathsModel,
            protected_media_root,
            validate_runtime_storage_contract,
        )

        for name in [
            "DATA_DIR_ENV",
            "PROTECTED_ROOT_ENV",
            "STORAGE_DIR_ENV",
            "PROTECTED_MEDIA_ROOT_ENV",
        ]:
            print(f"{name:<40} = {getattr(endoreg_env, name)}")

        paths = EndoregPathsModel.from_environment()
        protected_media = protected_media_root()

        print(f"endoreg protected_root       = {paths.protected_root}")
        print(f"endoreg data                 = {paths.data}")
        print(f"endoreg storage              = {paths.storage}")
        print(f"endoreg protected_media_root = {protected_media}")
        print(f"endoreg anonym_video         = {paths.anonym_video}")
        print(f"endoreg frame                = {paths.frame}")

        if (
            paths.protected_root.resolve()
            == Path(env("LX_ANNOTATE_ENCRYPTED_DATA_DIR")).resolve()
        ):
            ok("protected_root matches LX_ANNOTATE_ENCRYPTED_DATA_DIR")
        else:
            fail("protected_root does not match LX_ANNOTATE_ENCRYPTED_DATA_DIR")

        if paths.storage.resolve() == Path(env("STORAGE_DIR")).resolve():
            ok("storage matches STORAGE_DIR")
        else:
            fail("storage does not match STORAGE_DIR")

        if protected_media.resolve() == Path(env("PROTECTED_MEDIA_ROOT")).resolve():
            ok("protected_media_root() matches PROTECTED_MEDIA_ROOT")
        else:
            fail("protected_media_root() does not match PROTECTED_MEDIA_ROOT")

        validate_runtime_storage_contract()
        ok("validate_runtime_storage_contract() passed")

    except Exception as exc:
        fail(f"Endoreg path contract failed: {type(exc).__name__}: {exc}")


def parse_training_config() -> dict[str, Any]:
    section("6. TRAINING CONFIG AND LABELSET")

    config_path = Path(
        env("TRAINING_CONFIG_PATH")
        or "lx_ai/ai_model_config/train_sandbox_postgres.yaml"
    )

    print(f"training_config_path = {config_path}")

    if not config_path.exists():
        fail(f"Training config does not exist: {config_path}")
        return {}

    try:
        import yaml

        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        ok("Training YAML parsed")

        for key in [
            "dataset_uuid",
            "data_source",
            "labelset_id",
            "labelset_version",
            "labelset_version_to_train",
            "treat_unlabeled_as_negative",
            "model_key",
            "model_name",
            "device",
            "epochs",
        ]:
            value = find_key_recursive(data, {key})
            if value is not None:
                print(f"  {key}: {value}")

        return data

    except Exception as exc:
        fail(f"Training YAML parse failed: {type(exc).__name__}: {exc}")
        return {}


def verify_labelset(config: dict[str, Any], args: argparse.Namespace) -> None:
    labelset_id_raw = args.labelset_id or find_key_recursive(
        config,
        {"labelset_id", "label_set_id"},
    )
    labelset_version_raw = args.labelset_version or find_key_recursive(
        config,
        {"labelset_version_to_train", "labelset_version", "version"},
    )

    labelset_id = int(labelset_id_raw) if labelset_id_raw is not None else None
    labelset_version = (
        int(labelset_version_raw) if labelset_version_raw is not None else None
    )

    print(f"detected labelset_id      = {labelset_id}")
    print(f"detected labelset_version = {labelset_version}")

    print()
    print("Label-related models and row counts:")

    for model_name in [
        "Label",
        "LabelSet",
        "LabelType",
        "ImageClassificationLabel",
        "ImageClassificationLabelSet",
        "VideoSegmentationLabel",
        "VideoSegmentationLabelSet",
    ]:
        model = get_model_by_name(model_name)
        if model is None:
            continue

        try:
            print(
                f"  {model._meta.label:<55} "
                f"table={model._meta.db_table:<45} "
                f"rows={model.objects.count()}"
            )
        except Exception as exc:
            warn(f"Could not count {model_name}: {type(exc).__name__}: {exc}")

    if labelset_id is None or labelset_version is None:
        warn(
            "Could not detect labelset_id/version. Pass --labelset-id and --labelset-version."
        )
        return

    try:
        from lx_ai.utils.db_loader_for_model_input import load_labelset_from_postgres

        signature = inspect.signature(load_labelset_from_postgres)
        kwargs: dict[str, Any] = {}

        for name, param in signature.parameters.items():
            if name in {"labelset_id", "label_set_id", "labelset_id_to_train"}:
                kwargs[name] = labelset_id
            elif name in {"version", "labelset_version", "labelset_version_to_train"}:
                kwargs[name] = labelset_version
            elif name == "treat_unlabeled_as_negative":
                kwargs[name] = bool(
                    find_key_recursive(config, {"treat_unlabeled_as_negative"}) or False
                )
            elif param.default is inspect._empty:
                warn(f"Cannot map required labelset loader parameter: {name}")

        loaded = load_labelset_from_postgres(**kwargs)
        size = len(loaded) if hasattr(loaded, "__len__") else None

        if size == 0:
            fail(
                f"No labels found for labelset_id={labelset_id}, "
                f"version={labelset_version}"
            )
        else:
            ok(
                f"Labelset loader works for labelset_id={labelset_id}, "
                f"version={labelset_version}, size={size}"
            )

    except Exception as exc:
        fail(
            f"Labelset check failed for labelset_id={labelset_id}, "
            f"version={labelset_version}: {type(exc).__name__}: {exc}"
        )


def verify_db_training_objects(video_limit: int) -> list[int]:
    section("7. VIDEOFILE, FRAME, ANNOTATION COUNTS")

    video_count = model_count("VideoFile")
    frame_count = model_count("Frame")
    annotation_count = model_count("ImageClassificationAnnotation")

    print(f"VideoFile rows                       = {video_count}")
    print(f"Frame rows                           = {frame_count}")
    print(f"ImageClassificationAnnotation rows   = {annotation_count}")

    if video_count == 0:
        fail("No VideoFile rows found")
    if frame_count == 0:
        fail("No Frame rows found")
    if annotation_count == 0:
        fail("No ImageClassificationAnnotation rows found")

    section("8. VIDEO SOURCE RESOLUTION AND MEDIA FILES")

    sample_annotation_ids: list[int] = []

    VideoFile = get_model_by_name("VideoFile")
    ImageClassificationAnnotation = get_model_by_name("ImageClassificationAnnotation")

    if VideoFile is None:
        fail("VideoFile model unavailable")
        return sample_annotation_ids

    try:
        from endoreg_db.export.frames.export_frames_with_labels import (
            _resolve_processed_video_source_path,
        )
        from lx_ai.utils.endoregdb_encrypted_frame_bridge import (
            _field_file_is_encrypted,
        )

        videos = list(
            VideoFile.objects.exclude(processed_file="").order_by("pk")[:video_limit]
        )

        print(f"Checking first {len(videos)} processed videos")

        if not videos:
            fail("No videos with processed_file found")

        for video in videos:
            processed_file = getattr(video, "processed_file", None)
            processed_name = getattr(processed_file, "name", "") or ""

            print()
            print(f"video_id={video.pk}")
            print(f"  processed_file.name = {processed_name}")

            source_path = Path(_resolve_processed_video_source_path(video))
            print(f"  resolved source     = {source_path}")

            if source_path.exists():
                ok(f"Video {video.pk}: resolved source exists")
            else:
                fail(f"Video {video.pk}: resolved source does not exist: {source_path}")

            try:
                source_path.resolve().relative_to(
                    Path(env("PROTECTED_MEDIA_ROOT")).resolve()
                )
                ok(f"Video {video.pk}: source is inside PROTECTED_MEDIA_ROOT")
            except ValueError:
                fail(f"Video {video.pk}: source is outside PROTECTED_MEDIA_ROOT")

            encrypted = _field_file_is_encrypted(video, source_path)
            print(f"  encrypted detected  = {encrypted}")

    except Exception as exc:
        fail(f"Video source resolution failed: {type(exc).__name__}: {exc}")

    section("9. ANNOTATION → FRAME → VIDEO MAPPING CHECK")

    if ImageClassificationAnnotation is None:
        fail("ImageClassificationAnnotation model unavailable")
        return sample_annotation_ids

    try:
        annotations = list(
            ImageClassificationAnnotation.objects.select_related(
                "frame", "frame__video"
            )
            .filter(frame__isnull=False, frame__video__isnull=False)
            .exclude(frame__video__processed_file="")
            .order_by("pk")[:video_limit]
        )

        if not annotations:
            fail("No annotations with frame -> video -> processed_file found")

        output_root = Path(env("FRAME_MATERIALIZATION_OUTPUT_ROOT"))
        ext = env("LXAI_VERIFY_EXT", "jpg") or "jpg"

        for ann in annotations:
            frame = ann.frame
            video = frame.video
            sample_annotation_ids.append(int(ann.pk))

            expected_path = (
                output_root / f"video_{video.pk}" / f"frame_{frame.pk}.{ext}"
            )

            print()
            print(f"annotation_id={ann.pk}")
            print(f"  frame_pk                 = {frame.pk}")
            print(
                f"  frame_number             = {getattr(frame, 'frame_number', None)}"
            )
            print(f"  video_id                 = {video.pk}")
            print(f"  expected lx-ai output    = {expected_path}")
            print(f"  expected output exists   = {expected_path.exists()}")

        ok(
            "Mapping check completed. lx-ai frame output uses frame PK, not frame_number."
        )

    except Exception as exc:
        fail(f"Annotation mapping failed: {type(exc).__name__}: {exc}")

    return sample_annotation_ids


def optional_sample_extraction(
    args: argparse.Namespace, annotation_ids: list[int]
) -> None:
    section("10. OPTIONAL SAMPLE FRAME MATERIALIZATION")

    if not args.extract:
        print("Sample extraction disabled.")
        print()
        print("Enable with:")
        print("  --extract --extract-limit 2")
        return

    if not annotation_ids:
        fail("Cannot run sample extraction because no sample annotations were found")
        return

    selected = annotation_ids[: args.extract_limit]

    try:
        from lx_ai.utils.endoregdb_encrypted_frame_bridge import (
            materialize_frames_for_lxai_annotations,
        )

        print(f"annotation_ids = {selected}")
        print(f"output_root    = {env('FRAME_MATERIALIZATION_OUTPUT_ROOT')}")
        print(f"fps            = {args.fps}")
        print(f"ext            = {args.ext}")

        result = materialize_frames_for_lxai_annotations(
            annotation_ids=selected,
            output_root=env("FRAME_MATERIALIZATION_OUTPUT_ROOT"),
            fps=args.fps,
            ext=args.ext,
            overwrite=False,
        )

        for annotation_id in selected:
            path_value = result.get(annotation_id)

            if not path_value:
                fail(f"No materialized path returned for annotation_id={annotation_id}")
                continue

            path = Path(path_value)
            print(f"annotation_id={annotation_id} -> {path}")

            if path.exists():
                ok(f"Materialized frame exists for annotation_id={annotation_id}")
            else:
                fail(
                    f"Materialized frame missing for annotation_id={annotation_id}: {path}"
                )

    except Exception as exc:
        fail(f"Sample extraction failed: {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify lx-ai production runtime.")

    parser.add_argument(
        "--env-file",
        default=os.environ.get(
            "LXAI_VERIFY_ENV_FILE",
            "/var/endoreg-service-user/lx-ai/.env.systemd",
        ),
    )
    parser.add_argument("--labelset-id", type=int, default=None)
    parser.add_argument("--labelset-version", type=int, default=None)
    parser.add_argument(
        "--video-limit",
        type=int,
        default=int(os.environ.get("LXAI_VERIFY_VIDEO_LIMIT", "5")),
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        default=truthy(os.environ.get("LXAI_VERIFY_EXTRACT")),
    )
    parser.add_argument(
        "--extract-limit",
        type=int,
        default=int(os.environ.get("LXAI_VERIFY_EXTRACT_LIMIT", "2")),
    )
    parser.add_argument(
        "--fps", type=float, default=float(os.environ.get("LXAI_VERIFY_FPS", "50.0"))
    )
    parser.add_argument("--ext", default=os.environ.get("LXAI_VERIFY_EXT", "jpg"))

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    section("0. LOAD SERVICE ENVIRONMENT")

    env_file = Path(args.env_file)
    print(f"env file = {env_file}")

    load_env_file(env_file)
    load_db_password_from_file()

    working_dir = Path(env("WORKING_DIR", "/var/endoreg-service-user/lx-ai"))

    if working_dir.exists():
        os.chdir(working_dir)
        ok(f"working directory = {working_dir}")
    else:
        fail(f"WORKING_DIR does not exist: {working_dir}")

    verify_paths()
    verify_imports_before_django()
    setup_django()
    verify_imports_after_django()
    verify_endoreg_paths()

    config = parse_training_config()
    verify_labelset(config, args)

    sample_annotation_ids = verify_db_training_objects(args.video_limit)
    optional_sample_extraction(args, sample_annotation_ids)

    section("11. SUMMARY")

    if WARNINGS:
        print()
        print("WARNINGS:")
        for message in WARNINGS:
            print(f"  - {message}")

    if FAILURES:
        print()
        print("FAILURES:")
        for message in FAILURES:
            print(f"  - {message}")

        print()
        print("RESULT: FAIL")
        return 1

    print()
    if WARNINGS:
        print("RESULT: PASS WITH WARNINGS")
    else:
        print("RESULT: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
