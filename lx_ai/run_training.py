from pathlib import Path
import os

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

from django.conf import settings
from lx_ai.utils.path_diagnostics import print_runtime_path_diagnostics,validate_runtime_paths_for_training

from lx_ai.training.bucket_logic import build_bucket_key, compute_bucket
from lx_ai.training.bucket_snapshot import save_bucket_snapshot
from lx_ai.ai_model_config.config import TrainingConfig
from lx_ai.ai_model_training.trainer_gastronet_multilabel import (
    train_gastronet_multilabel,
)
from lx_ai.utils.logging_utils import section, subsection, kv
print("\n")
#print("Using database config:", settings.DATABASES)
#print("Data dir:", settings.DATA_DIR)
#print("Frame dir:", settings.FRAME_DIR)


def main() -> None:

    training_config_path = Path(
        os.getenv(
            "TRAINING_CONFIG_PATH",
            "lx_ai/ai_model_config/train_sandbox_postgres.yaml",
        )
    )
    
    print_runtime_path_diagnostics()
    
    cfg = TrainingConfig.from_yaml_file(training_config_path)
    
    validate_runtime_paths_for_training(cfg)
        

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

    out = train_gastronet_multilabel(cfg)

    subsection("ARTIFACTS")
    kv("Model saved to", out["model_path"])
    kv("Metadata saved to", out["meta_path"])

    section("TRAINING COMPLETE")


if __name__ == "__main__":
    main()