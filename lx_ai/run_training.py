from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

# =============================================================================
# DJANGO ENVIRONMENT SELECTION
# =============================================================================


def _default_django_settings_module() -> str:
    """
    Select Django settings according to the active runtime environment.

    Priority:
      1. Explicit DJANGO_SETTINGS_MODULE is respected by os.environ.setdefault.
      2. Otherwise DJANGO_ENV determines development or production.
    """

    environment = (
        os.getenv(
            "DJANGO_ENV",
            "development",
        )
        .strip()
        .lower()
    )

    if environment in {"production", "prod"}:
        return os.getenv(
            "DJANGO_SETTINGS_MODULE_PRODUCTION",
            "lx_ai.settings.settings_prod",
        )

    if environment in {"development", "dev", "local"}:
        return os.getenv(
            "DJANGO_SETTINGS_MODULE_DEVELOPMENT",
            "lx_ai.settings.settings_dev",
        )

    raise RuntimeError(
        f"Unsupported DJANGO_ENV: {environment!r}. Expected development or production."
    )


if not os.getenv("DJANGO_SETTINGS_MODULE", "").strip():
    os.environ["DJANGO_SETTINGS_MODULE"] = _default_django_settings_module()


# =============================================================================
# DJANGO INITIALIZATION
# =============================================================================

import django  # noqa: E402

django.setup()


# =============================================================================
# APPLICATION IMPORTS
# =============================================================================

from lx_ai.ai_model_config.config import TrainingConfig  # noqa: E402
from lx_ai.utils.logging_utils import (  # noqa: E402
    kv,
    section,
    subsection,
)
from lx_ai.utils.path_diagnostics import (  # noqa: E402
    print_runtime_path_diagnostics,
    validate_runtime_paths_for_training,
)


# =============================================================================
# TRAINER TYPE CONTRACT
# =============================================================================

TrainFunction = Callable[[TrainingConfig], dict[str, Any]]


# =============================================================================
# TRAINER LOADING
# =============================================================================


def _load_train_function() -> TrainFunction:
    """
    Import trainer lazily.

    This keeps run_training import lightweight for diagnostics and tests.
    Production behavior remains unchanged because main() loads the real
    trainer when no test trainer is provided.
    """

    from lx_ai.ai_model_training.trainer_gastronet_multilabel import (
        train_gastronet_multilabel,
    )

    return train_gastronet_multilabel


# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================


def _get_training_config_path() -> Path:
    """
    Resolve the shared training YAML.

    Development:
        Uses the repository-relative configuration path.

    Production:
        Uses the absolute path supplied by the NixOS service.

    Both environments use the same YAML file.
    """

    configured_path = os.getenv(
        "TRAINING_CONFIG_PATH",
        "lx_ai/ai_model_config/train_sandbox_postgres.yaml",
    )

    path = Path(configured_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Training configuration file does not exist: {path}")

    return path


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================


def main(
    train_fn: TrainFunction | None = None,
) -> None:
    # -------------------------------------------------------------------------
    # 1. Runtime environment
    # -------------------------------------------------------------------------

    environment = (
        os.getenv(
            "DJANGO_ENV",
            "development",
        )
        .strip()
        .lower()
    )

    settings_module = os.environ["DJANGO_SETTINGS_MODULE"]

    section("RUNTIME ENVIRONMENT")

    kv("Environment", environment)
    kv("Django settings", settings_module)
    kv("Working directory", Path.cwd())
    kv("Training configuration", os.getenv("TRAINING_CONFIG_PATH"))

    # -------------------------------------------------------------------------
    # 2. Runtime path diagnostics
    # -------------------------------------------------------------------------

    print_runtime_path_diagnostics()

    # -------------------------------------------------------------------------
    # 3. Load shared training configuration
    # -------------------------------------------------------------------------

    training_config_path = _get_training_config_path()

    cfg = TrainingConfig.from_yaml_file(training_config_path)
    kv("Training configuration", training_config_path)

    # -------------------------------------------------------------------------
    # 4. Validate runtime paths
    # -------------------------------------------------------------------------

    validate_runtime_paths_for_training(cfg)

    # -------------------------------------------------------------------------
    # 5. Load trainer
    # -------------------------------------------------------------------------

    if train_fn is None:
        train_fn = _load_train_function()

    # -------------------------------------------------------------------------
    # 6. Training start
    # -------------------------------------------------------------------------

    section("TRAINING START")

    subsection("CONFIG")

    kv("Dataset UUID", cfg.dataset_uuid)
    kv("Data source", cfg.data_source)

    kv(
        "Labelset",
        f"id={cfg.labelset_id}, version={cfg.labelset_version_to_train}",
    )

    kv(
        "Treat unlabeled as neg",
        cfg.treat_unlabeled_as_negative,
    )

    kv("Model selected", cfg.backbone_name)
    kv("Device", cfg.device)
    kv("Seed", cfg.random_seed)
    kv("Epochs", cfg.num_epochs)
    kv("Backbone checkpoint", cfg.backbone_checkpoint)
    kv("Total buckets", cfg.bucket_policy.num_buckets)

    # -------------------------------------------------------------------------
    # 7. Execute training
    # -------------------------------------------------------------------------

    out = train_fn(cfg)

    # -------------------------------------------------------------------------
    # 8. Training artifacts
    # -------------------------------------------------------------------------

    subsection("ARTIFACTS")

    kv("Model saved to", out["model_path"])
    kv("Metadata saved to", out["meta_path"])

    # -------------------------------------------------------------------------
    # 9. Completion
    # -------------------------------------------------------------------------

    section("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
