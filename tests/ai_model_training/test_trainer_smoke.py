"""
Trainer smoke test.

This test checks that the full training pipeline can run end to end
without crashing.

What this test is doing:

- Create two small real images
- Create two JSONL entries
- Build TrainingConfig with minimal settings
- Run training for 1 epoch on CPU
- Check that model file and meta file are created

Why two samples:

Trainer always creates train and validation datasets.
So at least one sample must exist for validation.

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
    GIVEN a minimal valid training configuration
    WHEN train_gastronet_multilabel is executed
    THEN training finishes and model artifacts are written

    This test uses:
    - 2 images
    - 1 epoch
    - CPU only
    """
    # ------------------------------------------------------------
    # 1. Create image directory
    # ------------------------------------------------------------
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    # ------------------------------------------------------------
    # 2. Create two real images
    # ------------------------------------------------------------
    img1 = image_dir / "img1.jpg"
    img2 = image_dir / "img2.jpg"

    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(img1)
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(img2)

    # ------------------------------------------------------------
    # 3. Create minimal JSONL file with two entries
    # ------------------------------------------------------------
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        (
            '{"labels": [], "old_examination_id": 1, "old_id": 1, "filename": "img1.jpg"}\n'
            '{"labels": [], "old_examination_id": 2, "old_id": 2, "filename": "img2.jpg"}\n'
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # 4. Patch default loader paths
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
