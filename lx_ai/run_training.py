from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Ensure Django settings module exists before setup
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    os.getenv(
        "DJANGO_SETTINGS_MODULE_DEVELOPMENT",
        "lx_ai.settings.settings_dev",
    ),
)

import django

django.setup()

from lx_ai.ai_model_config.config import TrainingConfig
from lx_ai.utils.logging_utils import kv, section, subsection
from lx_ai.utils.path_diagnostics import (
    print_runtime_path_diagnostics,
    validate_runtime_paths_for_training,
)


TrainFunction = Callable[[TrainingConfig], dict[str, Any]]


def _load_train_function() -> TrainFunction:
    """
    Import trainer lazily.

    This keeps run_training import lightweight for diagnostics and tests.
    Production behavior remains the same because main() still loads the real trainer
    when no test trainer is provided.
    """
    from lx_ai.ai_model_training.trainer_gastronet_multilabel import (
        train_gastronet_multilabel,
    )

    return train_gastronet_multilabel


def main(train_fn: TrainFunction | None = None) -> None:
    training_config_path = Path(
        os.getenv(
            "TRAINING_CONFIG_PATH",
            "lx_ai/ai_model_config/train_sandbox_postgres.yaml",
        )
    )

    print_runtime_path_diagnostics()

    cfg = TrainingConfig.from_yaml_file(training_config_path)

    validate_runtime_paths_for_training(cfg)

    if train_fn is None:
        train_fn = _load_train_function()

    section("TRAINING START")

    subsection("CONFIG")
    kv("Dataset UUID", cfg.dataset_uuid)
    kv("Data source", cfg.data_source)
    kv("Labelset", f"id={cfg.labelset_id}, version={cfg.labelset_version_to_train}")
    kv("Treat unlabeled as neg", cfg.treat_unlabeled_as_negative)
    kv("Model selected", cfg.backbone_name)
    kv("Device", cfg.device)
    kv("Seed", cfg.random_seed)
    kv("Epochs", cfg.num_epochs)
    kv("Backbone checkpoint", cfg.backbone_checkpoint)
    kv("Total buckets", cfg.bucket_policy.num_buckets)

    out = train_fn(cfg)

    subsection("ARTIFACTS")
    kv("Model saved to", out["model_path"])
    kv("Metadata saved to", out["meta_path"])

    section("TRAINING COMPLETE")


if __name__ == "__main__":
    main()