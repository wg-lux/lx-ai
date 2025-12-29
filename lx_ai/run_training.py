from lx_ai.ai_model_config.config import TrainingConfig
from lx_ai.ai_model_training.trainer_gastronet_multilabel import train_gastronet_multilabel
from pathlib import Path

def main() -> None:
    cfg = TrainingConfig(
        dataset_uuid="legacy_jsonl_v1",
        labelset_version_to_train=2,
        treat_unlabeled_as_negative=True,   # matches your loader run

        backbone_name="gastro_rn50",
        backbone_checkpoint=Path("/home/admin/dev/lx-ai/data/model_training/checkpoints/RN50_GastroNet-1M_DINOv1.pth"),
        freeze_backbone=True,

        num_epochs=5,
        batch_size=16,
        val_split=0.2,
        test_split=0.1,

        lr_head=1e-3,
        lr_backbone=1e-4,
        gamma_focal=2.0,
        alpha_focal=0.25,

        use_scheduler=True,
        warmup_epochs=2,
        min_lr=1e-6,

        device="cuda",
        random_seed=42,
    )

    out = train_gastronet_multilabel(cfg)
    print("DONE")
    print("Model:", out["model_path"])
    print("Meta :", out["meta_path"])


if __name__ == "__main__":
    main()
