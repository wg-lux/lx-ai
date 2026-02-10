from pathlib import Path

from lx_ai.ai_model_config.config import TrainingConfig
from lx_ai.ai_model_training.trainer_gastronet_multilabel import (
    train_gastronet_multilabel,
)
from lx_ai.utils.logging_utils import section, subsection


def main() -> None:
    # -------------------------------------------------------------------------
    # Load config (single source of truth)
    # -------------------------------------------------------------------------
    cfg = TrainingConfig.from_yaml_file(
        Path("lx_ai/ai_model_config/train_sandbox_postgres.yaml")
    )
    
    # -------------------------------------------------------------------------
    # Training start banner
    # -------------------------------------------------------------------------
    section("TRAINING START")

    subsection("CONFIG")
    print(f"  Dataset UUID          : {cfg.dataset_uuid}")
    print(f"  Data source           : {cfg.data_source}")
    print(f"  Labelset              : id={cfg.labelset_id}, version={cfg.labelset_version_to_train}")
    print(f"  Treat unlabeled as neg: {cfg.treat_unlabeled_as_negative}")
    print(f"  Device                : {cfg.device}")
    print(f"  Seed                  : {cfg.random_seed}")

    # -------------------------------------------------------------------------
    # Run training (ALL heavy logs happen inside trainer)
    # -------------------------------------------------------------------------
    out = train_gastronet_multilabel(cfg)

    # -------------------------------------------------------------------------
    # Artifacts summary
    # -------------------------------------------------------------------------
    subsection("ARTIFACTS")
    print(f"  Model saved to        : {out['model_path']}")
    print(f"  Metadata saved to     : {out['meta_path']}")

    # -------------------------------------------------------------------------
    # Training end banner
    # -------------------------------------------------------------------------
    section("TRAINING COMPLETE")
if __name__ == "__main__":
    main()
