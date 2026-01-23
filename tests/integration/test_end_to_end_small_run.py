"""
Integration test: very small end-to-end training run.

This test ensures the full training pipeline works with
a minimal but valid dataset.

What this test is doing:
- Create three real images
- Create three JSONL entries
- Run full training pipeline
- Verify output artifacts exist

This test checks:
- data loading
- dataset splitting
- training loop execution
- artifact writing

This test does NOT check:
- accuracy
- convergence
- metric values
"""

from pathlib import Path

import pytest
from PIL import Image

from lx_ai.ai_model_config.config import TrainingConfig
from lx_ai.ai_model_training.trainer_gastronet_multilabel import (
    train_gastronet_multilabel,
)


@pytest.mark.localfiles
def test_end_to_end_small_run(tmp_path: Path, monkeypatch) -> None:
    """
    GIVEN a very small but valid dataset
    WHEN full training pipeline is executed
    THEN training completes and output files exist
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
        Image.new("RGB", (32, 32), color=(i * 30, i * 30, i * 30)).save(img)

    # ------------------------------------------------------------
    # 3. Create JSONL file
    # ------------------------------------------------------------
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        (
            '{"labels": [], "old_examination_id": 10, "old_id": 1, "filename": "img1.jpg"}\n'
            '{"labels": [], "old_examination_id": 20, "old_id": 2, "filename": "img2.jpg"}\n'
            '{"labels": [], "old_examination_id": 30, "old_id": 3, "filename": "img3.jpg"}\n'
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
        dataset_uuid="integration_small_run",
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
