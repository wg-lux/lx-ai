from pathlib import Path

import torch

from lx_ai.ai_model_dataset.dataset import (
    EndoMultiLabelDataset,
    MultiLabelDatasetSpec,
)

from PIL import Image


# -----------------------------------------------------------------------------
# Smoke test for torch Dataset
# -----------------------------------------------------------------------------

def test_torch_dataset__basic_iteration__works(tmp_path: Path) -> None:
    """
    GIVEN a valid MultiLabelDatasetSpec
    WHEN wrapped in EndoMultiLabelDataset
    THEN __len__ and __getitem__ work and return tensors
    """

    img = tmp_path / "img.jpg"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(img)


    spec = MultiLabelDatasetSpec(
        image_paths=[img],
        label_vectors=[[1, None]],
        label_masks=[[1, 0]],
        image_size=64,
    )

    ds = EndoMultiLabelDataset(spec)

    assert len(ds) == 1

    x, y, m = ds[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(m, torch.Tensor)

    assert x.shape == (3, 64, 64)
    assert y.shape == (2,)
    assert m.shape == (2,)
