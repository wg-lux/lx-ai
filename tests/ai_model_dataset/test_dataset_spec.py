from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from lx_ai.ai_model_dataset.dataset import EndoMultiLabelDataset, MultiLabelDatasetSpec


def _make_image(path: Path, size: tuple[int, int] = (32, 32)) -> Path:
    # creates a small valid rgb image for dataset image loading tests
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(120, 80, 40))
    img.save(path)
    return path


def _valid_spec_data(tmp_path: Path) -> dict:
    # base valid dataset data used in many tests
    img1 = _make_image(tmp_path / "images" / "frame_1.jpg")
    img2 = _make_image(tmp_path / "images" / "frame_2.jpg")

    return {
        "image_paths": [img1, img2],
        "label_vectors": [
            [1, 0, None],
            [0, 1, None],
        ],
        "label_masks": [
            [1, 1, 0],
            [1, 1, 0],
        ],
        "image_size": 64,
    }


def test_valid_dataset_spec_passes(tmp_path: Path) -> None:
    # checks that a valid dataset spec is accepted
    spec = MultiLabelDatasetSpec.model_validate(_valid_spec_data(tmp_path))

    assert len(spec.image_paths) == 2
    assert len(spec.label_vectors) == 2
    assert len(spec.label_masks) == 2
    assert spec.image_size == 64


def test_image_paths_are_converted_from_strings(tmp_path: Path) -> None:
    # checks that image path strings are converted to Path objects
    data = _valid_spec_data(tmp_path)
    data["image_paths"] = [str(p) for p in data["image_paths"]]

    spec = MultiLabelDatasetSpec.model_validate(data)

    assert all(isinstance(p, Path) for p in spec.image_paths)


def test_image_paths_must_be_list(tmp_path: Path) -> None:
    # checks image_paths must be a list
    data = _valid_spec_data(tmp_path)
    data["image_paths"] = "not-a-list"

    with pytest.raises(TypeError, match="image_paths must be a list"):
        MultiLabelDatasetSpec.model_validate(data)


def test_image_paths_items_must_be_path_or_string(tmp_path: Path) -> None:
    # checks image_paths items must be string or Path
    data = _valid_spec_data(tmp_path)
    data["image_paths"] = [123]

    with pytest.raises(TypeError, match="image_paths items must be str\\|Path"):
        MultiLabelDatasetSpec.model_validate(data)


def test_label_vectors_reject_invalid_value(tmp_path: Path) -> None:
    # checks label vectors only allow 0, 1 or None
    data = _valid_spec_data(tmp_path)
    data["label_vectors"] = [
        [1, 2, None],
        [0, 1, None],
    ]

    with pytest.raises(ValidationError, match="must be 0\\|1\\|None"):
        MultiLabelDatasetSpec.model_validate(data)


def test_label_masks_reject_invalid_value(tmp_path: Path) -> None:
    # checks label masks only allow 0 or 1
    data = _valid_spec_data(tmp_path)
    data["label_masks"] = [
        [1, 2, 0],
        [1, 1, 0],
    ]

    with pytest.raises(ValidationError, match="must be 0\\|1"):
        MultiLabelDatasetSpec.model_validate(data)


def test_image_vectors_and_masks_must_have_same_number_of_samples(
    tmp_path: Path,
) -> None:
    # checks image_paths, label_vectors and label_masks must align by sample count
    data = _valid_spec_data(tmp_path)
    data["label_vectors"] = [[1, 0, None]]

    with pytest.raises(ValidationError, match="must have the same length"):
        MultiLabelDatasetSpec.model_validate(data)


def test_label_vectors_must_have_same_label_count(tmp_path: Path) -> None:
    # checks all label vectors must have same number of labels
    data = _valid_spec_data(tmp_path)
    data["label_vectors"] = [
        [1, 0, None],
        [0, 1],
    ]

    with pytest.raises(ValidationError, match="label_vectors\\[1\\] length mismatch"):
        MultiLabelDatasetSpec.model_validate(data)


def test_label_masks_must_have_same_label_count(tmp_path: Path) -> None:
    # checks all label masks must have same number of labels
    data = _valid_spec_data(tmp_path)
    data["label_masks"] = [
        [1, 1, 0],
        [1, 1],
    ]

    with pytest.raises(ValidationError, match="label_masks\\[1\\] length mismatch"):
        MultiLabelDatasetSpec.model_validate(data)


def test_none_label_requires_mask_zero(tmp_path: Path) -> None:
    # checks None label must have mask 0 because it means unknown
    data = _valid_spec_data(tmp_path)
    data["label_vectors"] = [
        [1, 0, None],
        [0, 1, None],
    ]
    data["label_masks"] = [
        [1, 1, 1],
        [1, 1, 0],
    ]

    with pytest.raises(ValidationError, match="value=None but mask=1"):
        MultiLabelDatasetSpec.model_validate(data)


def test_known_label_requires_mask_one(tmp_path: Path) -> None:
    # checks known label value 0 or 1 must have mask 1
    data = _valid_spec_data(tmp_path)
    data["label_vectors"] = [
        [1, 0, None],
        [0, 1, None],
    ]
    data["label_masks"] = [
        [1, 0, 0],
        [1, 1, 0],
    ]

    with pytest.raises(ValidationError, match="value=0 but mask=0"):
        MultiLabelDatasetSpec.model_validate(data)


def test_image_size_must_be_at_least_16(tmp_path: Path) -> None:
    # checks image_size must be valid and not too small
    data = _valid_spec_data(tmp_path)
    data["image_size"] = 8

    with pytest.raises(ValidationError):
        MultiLabelDatasetSpec.model_validate(data)


def test_to_ddict_returns_json_safe_values(tmp_path: Path) -> None:
    # checks to_ddict returns strings for paths and plain python values
    spec = MultiLabelDatasetSpec.model_validate(_valid_spec_data(tmp_path))

    ddict = spec.to_ddict()

    assert isinstance(ddict["image_paths"][0], str)
    assert ddict["label_vectors"] == [
        [1, 0, None],
        [0, 1, None],
    ]
    assert ddict["label_masks"] == [
        [1, 1, 0],
        [1, 1, 0],
    ]
    assert ddict["image_size"] == 64


def test_torch_dataset_len_matches_number_of_images(tmp_path: Path) -> None:
    # checks torch dataset length equals number of image paths
    spec = MultiLabelDatasetSpec.model_validate(_valid_spec_data(tmp_path))
    ds = EndoMultiLabelDataset(spec)

    assert len(ds) == 2


def test_torch_dataset_converts_none_labels_to_zero_with_mask_zero(
    tmp_path: Path,
) -> None:
    # checks None labels become 0 in tensor but mask stays 0
    spec = MultiLabelDatasetSpec.model_validate(_valid_spec_data(tmp_path))
    ds = EndoMultiLabelDataset(spec)

    assert torch.equal(
        ds.labels,
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )

    assert torch.equal(
        ds.masks,
        torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_torch_dataset_getitem_returns_tensor_label_and_mask(tmp_path: Path) -> None:
    # checks __getitem__ returns image tensor, label tensor and mask tensor
    spec = MultiLabelDatasetSpec.model_validate(_valid_spec_data(tmp_path))
    ds = EndoMultiLabelDataset(spec)

    x, y, m = ds[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(m, torch.Tensor)

    assert x.shape == (3, 64, 64)
    assert y.shape == (3,)
    assert m.shape == (3,)

    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert m.dtype == torch.float32


def test_torch_dataset_image_is_normalized(tmp_path: Path) -> None:
    # checks loaded image is converted to normalized tensor
    spec = MultiLabelDatasetSpec.model_validate(_valid_spec_data(tmp_path))
    ds = EndoMultiLabelDataset(spec)

    x, _, _ = ds[0]

    assert x.shape == (3, 64, 64)
    assert torch.isfinite(x).all()
    assert not torch.equal(x, torch.zeros_like(x))


def test_torch_dataset_missing_image_raises_file_error(tmp_path: Path) -> None:
    # checks missing image file raises error when sample is loaded
    data = _valid_spec_data(tmp_path)
    missing = tmp_path / "images" / "missing.jpg"
    data["image_paths"] = [missing, data["image_paths"][1]]

    spec = MultiLabelDatasetSpec.model_validate(data)
    ds = EndoMultiLabelDataset(spec)

    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_empty_dataset_is_rejected(tmp_path: Path) -> None:
    # checks empty dataset is not allowed
    data = {
        "image_paths": [],
        "label_vectors": [],
        "label_masks": [],
        "image_size": 64,
    }

    with pytest.raises(ValidationError):
        MultiLabelDatasetSpec.model_validate(data)


def test_zero_label_columns_are_rejected(tmp_path: Path) -> None:
    # checks label vectors must contain at least one label column
    img1 = _make_image(tmp_path / "images" / "frame_1.jpg")

    data = {
        "image_paths": [img1],
        "label_vectors": [[]],
        "label_masks": [[]],
        "image_size": 64,
    }

    with pytest.raises(ValidationError, match="at least one label"):
        MultiLabelDatasetSpec.model_validate(data)
