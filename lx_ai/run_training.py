# lx_ai/run_training.py

from pathlib import Path

from lx_ai.ai_model_config.config import TrainingConfig
from lx_ai.ai_model_training.trainer_gastronet_multilabel import train_gastronet_multilabel


def main() -> None:
    # Load config from YAML (single source of truth)
    cfg = TrainingConfig.from_yaml_file(
        Path("lx_ai/ai_model_config/train_sandbox_postgres.yaml")
)


    out = train_gastronet_multilabel(cfg)

    print("DONE")
    print("Model:", out["model_path"])
    print("Meta :", out["meta_path"])


if __name__ == "__main__":
    main()
