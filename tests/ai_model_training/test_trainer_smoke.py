"""
Trainer smoke test.

This test checks that the full training pipeline can run end to end
without crashing.

What this test is doing:

- Create three small real images
- Create three JSONL entries with different examination ids
- Build TrainingConfig with validation enabled (default)
- Run training for 1 epoch on CPU
- Check that model file and meta file are created

Why three samples:

Trainer always creates a validation dataset.
With group-wise split, at least one group must go to validation.

What this test does NOT check:
- accuracy
- learning quality
- convergence

This test only checks pipeline stability.
"""

from pathlib import Path

import pytest
from PIL import Image

from lx_ai.ai_model_config.config import TrainingConfig
from lx_ai.ai_model_training.trainer_gastronet_multilabel import (
    train_gastronet_multilabel,
)


@pytest.mark.localfiles
def test_trainer_smoke__end_to_end_run(tmp_path: Path, monkeypatch) -> None:
    """
    GIVEN a minimal valid training configuration with validation enabled
    WHEN train_gastronet_multilabel is executed
    THEN training finishes and model artifacts are written
    """

    # ------------------------------------------------------------
    # 1. Create image directory
    # ------------------------------------------------------------
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    # ------------------------------------------------------------
    # 2. Create three real images
    # ------------------------------------------------------------
    for i in range(1, 4):
        img = image_dir / f"img{i}.jpg"
        Image.new("RGB", (16, 16), color=(i * 40, i * 40, i * 40)).save(img)

    # ------------------------------------------------------------
    # 3. Create JSONL file with three entries
    # ------------------------------------------------------------
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        (
            '{"labels": [], "old_examination_id": 1, "old_id": 1, "filename": "img1.jpg"}\n'
            '{"labels": [], "old_examination_id": 2, "old_id": 2, "filename": "img2.jpg"}\n'
            '{"labels": [], "old_examination_id": 3, "old_id": 3, "filename": "img3.jpg"}\n'
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # 4. Patch loader defaults
    # ------------------------------------------------------------
    monkeypatch.setattr(
        "lx_ai.utils.data_loader_for_model_input.DEFAULT_IMAGE_DIR",
        image_dir,
    )
    monkeypatch.setattr(
        "lx_ai.utils.data_loader_for_model_input.DEFAULT_JSONL_PATH",
        jsonl_path,
    )

    # ------------------------------------------------------------
    # 5. Build training configuration
    # ------------------------------------------------------------
    cfg = TrainingConfig(
        dataset_uuid="smoke_ds",
        data_source="jsonl",
        jsonl_path=jsonl_path,
        base_dir=tmp_path,
        num_epochs=1,
        batch_size=1,
        device="cpu",
        val_split=0.34,
        test_split=0.33,
        create_dirs=True,
)


    # ------------------------------------------------------------
    # 6. Run training
    # ------------------------------------------------------------
    out = train_gastronet_multilabel(cfg)

    # ------------------------------------------------------------
    # 7. Validate outputs
    # ------------------------------------------------------------
    model_path = Path(out["model_path"])
    meta_path = Path(out["meta_path"])

    assert model_path.exists()
    assert meta_path.exists()
    assert model_path.suffix == ".pth"
    assert meta_path.suffix == ".json"
