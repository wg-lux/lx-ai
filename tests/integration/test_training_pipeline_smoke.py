from __future__ import annotations

from pathlib import Path

from lx_ai import run_training


def test_training_pipeline_smoke_executes_from_config_to_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # checks the full training entrypoint wiring executes without real DB or GPU

    data_dir = tmp_path / "data"
    conf_dir = tmp_path / "conf"
    training_root = data_dir / "model_training"
    checkpoints_dir = training_root / "checkpoints"
    runs_dir = training_root / "runs"
    buckets_dir = training_root / "buckets"

    data_dir.mkdir(parents=True)
    conf_dir.mkdir(parents=True)
    checkpoints_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)
    buckets_dir.mkdir(parents=True)

    checkpoint = checkpoints_dir / "RN50_GastroNet-1M_DINOv1.pth"
    checkpoint.write_bytes(b"fake-checkpoint")

    sqlite_db = tmp_path / "dev_db.sqlite"
    sqlite_db.write_text("", encoding="utf-8")

    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        f"""
dataset_uuid: smoke_ds
data_source: postgres
dataset_ids: [1]
labelset_id: 5
labelset_version_to_train: 3
treat_unlabeled_as_negative: false

base_dir: "{data_dir}"
training_root: "{training_root}"
checkpoints_dir: "{checkpoints_dir}"
runs_dir: "{runs_dir}"
create_dirs: true

backbone_name: gastro_rn50
backbone_checkpoint: "{checkpoint}"
freeze_backbone: true

num_epochs: 1
batch_size: 2

lr_head: 0.001
lr_backbone: 0.0001
gamma_focal: 2.0
alpha_focal: 0.25

use_scheduler: false
warmup_epochs: 0
min_lr: 1.0e-6

device: cpu
random_seed: 42

bucket_policy:
  num_buckets: 5
  validation_buckets: [3]
  test_buckets: [4]

save_bucket_snapshot: false
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("CONF_DIR", str(conf_dir))
    monkeypatch.setenv("TRAINING_ROOT", str(training_root))
    monkeypatch.setenv("CHECKPOINTS_DIR", str(checkpoints_dir))
    monkeypatch.setenv("RUNS_DIR", str(runs_dir))
    monkeypatch.setenv("BUCKET_SNAPSHOT_DIR", str(buckets_dir))
    monkeypatch.setenv("BACKBONE_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("TRAINING_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(sqlite_db))

    def fake_train_gastronet_multilabel(config):
        # simulates successful trainer output without touching DB, GPU, or model code
        model_path = Path(config.runs_dir) / "smoke_model.pth"
        meta_path = Path(config.runs_dir) / "smoke_model_meta.json"

        model_path.write_bytes(b"fake-model")
        meta_path.write_text('{"ok": true}', encoding="utf-8")

        return {
            "model_path": str(model_path),
            "meta_path": str(meta_path),
            "history": {
                "train_loss": [0.1],
                "val_loss": [None],
                "test_loss": None,
            },
        }

    run_training.main(train_fn=fake_train_gastronet_multilabel)

    assert (runs_dir / "smoke_model.pth").exists()
    assert (runs_dir / "smoke_model_meta.json").exists()
