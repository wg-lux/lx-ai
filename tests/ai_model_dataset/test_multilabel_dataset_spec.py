from pathlib import Path

import pytest

from lx_ai.ai_model_dataset.dataset import MultiLabelDatasetSpec


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def minimal_valid_spec(tmp_path: Path) -> dict:
    """
    Minimal valid dataset specification.

    Represents the smallest dataset that should pass validation.
    """
    img = tmp_path / "img.jpg"
    img.write_bytes(b"fake-image")

    return {
        "image_paths": [img],
        "label_vectors": [[1, None]],
        "label_masks": [[1, 0]],
        "image_size": 224,
    }


# -----------------------------------------------------------------------------
# 1. Minimal valid spec
# -----------------------------------------------------------------------------

def test_dataset_spec__minimal_valid__passes(tmp_path: Path) -> None:
    """
    GIVEN a minimal valid dataset spec
    WHEN MultiLabelDatasetSpec is created
    THEN validation succeeds
    """
    spec = MultiLabelDatasetSpec(**minimal_valid_spec(tmp_path))

    assert len(spec.image_paths) == 1
    assert spec.label_vectors[0] == [1, None]
    assert spec.label_masks[0] == [1, 0]


# -----------------------------------------------------------------------------
# 2. Length alignment validation
# -----------------------------------------------------------------------------

def test_dataset_spec__length_mismatch__fails(tmp_path: Path) -> None:
    """
    image_paths, label_vectors and label_masks must have same length.
    """
    data = minimal_valid_spec(tmp_path)
    data["label_vectors"].append([0, 1])

    with pytest.raises(ValueError, match="must have the same length"):
        MultiLabelDatasetSpec(**data)


# -----------------------------------------------------------------------------
# 3. Label value validation
# -----------------------------------------------------------------------------

def test_dataset_spec__invalid_label_value__fails(tmp_path: Path) -> None:
    """
    label_vectors values must be 0, 1 or None.
    """
    data = minimal_valid_spec(tmp_path)
    data["label_vectors"] = [[2, None]]

    with pytest.raises(ValueError, match="must be 0\\|1\\|None"):
        MultiLabelDatasetSpec(**data)


# -----------------------------------------------------------------------------
# 4. Mask value validation
# -----------------------------------------------------------------------------

def test_dataset_spec__invalid_mask_value__fails(tmp_path: Path) -> None:
    """
    label_masks values must be 0 or 1.
    """
    data = minimal_valid_spec(tmp_path)
    data["label_masks"] = [[1, 2]]

    with pytest.raises(ValueError, match="must be 0\\|1"):
        MultiLabelDatasetSpec(**data)


# -----------------------------------------------------------------------------
# 5. Mask / value semantic consistency
# -----------------------------------------------------------------------------

def test_dataset_spec__unknown_label_with_mask_one__fails(tmp_path: Path) -> None:
    """
    Unknown labels (None) must have mask=0.
    """
    data = minimal_valid_spec(tmp_path)
    data["label_masks"] = [[1, 1]]

    with pytest.raises(ValueError, match="Inconsistent unknown label"):
        MultiLabelDatasetSpec(**data)


def test_dataset_spec__known_label_with_mask_zero__fails(tmp_path: Path) -> None:
    """
    Known labels (0 or 1) must have mask=1.
    """
    data = minimal_valid_spec(tmp_path)
    data["label_vectors"] = [[1, 0]]
    data["label_masks"] = [[1, 0]]

    with pytest.raises(ValueError, match="Inconsistent known label"):
        MultiLabelDatasetSpec(**data)
