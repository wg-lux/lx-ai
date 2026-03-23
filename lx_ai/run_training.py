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

from lx_ai.training.bucket_logic import build_bucket_key, compute_bucket
from lx_ai.training.bucket_snapshot import save_bucket_snapshot
from lx_ai.ai_model_config.config import TrainingConfig
from lx_ai.ai_model_training.trainer_gastronet_multilabel import (
    train_gastronet_multilabel,
)
from lx_ai.utils.logging_utils import section, subsection


print("Using database config:", settings.DATABASES)
print("Data dir:", settings.DATA_DIR)
print("Frame dir:", settings.FRAME_DIR)


def main() -> None:

    cfg = TrainingConfig.from_yaml_file(
        Path("lx_ai/ai_model_config/train_sandbox_postgres.yaml")
    )

    section("TRAINING START")

    subsection("CONFIG")
    print(f"  Dataset UUID          : {cfg.dataset_uuid}")
    print(f"  Data source           : {cfg.data_source}")
    print(
        f"  Labelset              : id={cfg.labelset_id}, version={cfg.labelset_version_to_train}"
    )
    print(f"  Treat unlabeled as neg: {cfg.treat_unlabeled_as_negative}")
    print(f"  Model Selected        : {cfg.backbone_name}")
    print(f"  Device                : {cfg.device}")
    print(f"  Seed                  : {cfg.random_seed}")
    print(f"  Epochs                : {cfg.num_epochs}")
    print(f"  Backbone checkpoint   : {cfg.backbone_checkpoint}")
    print(f"  Total Buckets         : {cfg.bucket_policy.num_buckets}")

    out = train_gastronet_multilabel(cfg)

    subsection("ARTIFACTS")
    print(f"  Model saved to        : {out['model_path']}")
    print(f"  Metadata saved to     : {out['meta_path']}")

    section("TRAINING COMPLETE")


if __name__ == "__main__":
    main()