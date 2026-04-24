from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from lx_ai.ai_model.losses import (
    FocalLossConfig,
    compute_class_weights,
    focal_loss_with_mask,
)


def test_focal_loss_config_defaults_are_valid() -> None:
    # checks default focal loss config values are valid
    cfg = FocalLossConfig()

    assert cfg.alpha == 0.25
    assert cfg.gamma == 2.0
    assert cfg.eps == 1e-6
    assert cfg.use_class_weights is True


def test_focal_loss_config_rejects_invalid_alpha() -> None:
    # checks alpha must be between 0 and 1
    with pytest.raises(ValidationError):
        FocalLossConfig(alpha=1.5)


def test_focal_loss_config_rejects_invalid_gamma() -> None:
    # checks gamma must not be negative
    with pytest.raises(ValidationError):
        FocalLossConfig(gamma=-1.0)


def test_focal_loss_config_rejects_too_large_eps() -> None:
    # checks eps must be small because large eps damages probabilities
    with pytest.raises(ValidationError, match="eps is too large"):
        FocalLossConfig(eps=0.2)


def test_focal_loss_config_to_ddict_returns_plain_values() -> None:
    # checks config can be converted to simple dict for metadata
    cfg = FocalLossConfig(alpha=0.3, gamma=1.5, eps=1e-5, use_class_weights=False)

    ddict = cfg.to_ddict()

    assert ddict == {
        "alpha": 0.3,
        "gamma": 1.5,
        "eps": 1e-5,
        "use_class_weights": False,
    }


def test_compute_class_weights_returns_one_weight_per_label() -> None:
    # checks class weights shape is one weight per label
    labels = torch.tensor(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=torch.float32,
    )
    masks = torch.ones_like(labels)

    weights = compute_class_weights(labels, masks)

    assert weights.shape == (3,)
    assert torch.isfinite(weights).all()
    assert torch.all(weights > 0)


def test_compute_class_weights_mean_is_close_to_one() -> None:
    # checks class weights are normalized so mean is close to 1
    labels = torch.tensor(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    masks = torch.ones_like(labels)

    weights = compute_class_weights(labels, masks)

    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-5)


def test_compute_class_weights_ignores_unknown_labels_by_mask() -> None:
    # checks masked unknown labels do not count as positives
    labels = torch.tensor(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=torch.float32,
    )
    masks = torch.tensor(
        [
            [1, 0],
            [1, 0],
        ],
        dtype=torch.float32,
    )

    weights = compute_class_weights(labels, masks)

    assert weights.shape == (2,)
    assert torch.isfinite(weights).all()
    assert torch.all(weights > 0)


def test_compute_class_weights_rejects_non_2d_labels() -> None:
    # checks labels must be 2D
    labels = torch.tensor([1, 0, 1], dtype=torch.float32)
    masks = torch.tensor([1, 1, 1], dtype=torch.float32)

    with pytest.raises(ValueError, match="labels must be 2D"):
        compute_class_weights(labels, masks)


def test_compute_class_weights_rejects_shape_mismatch() -> None:
    # checks labels and masks must have same shape
    labels = torch.zeros((2, 3), dtype=torch.float32)
    masks = torch.zeros((2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="masks must match labels shape"):
        compute_class_weights(labels, masks)


def test_compute_class_weights_rejects_empty_tensor() -> None:
    # checks empty labels are not allowed
    labels = torch.empty((0, 3), dtype=torch.float32)
    masks = torch.empty((0, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="labels must not be empty"):
        compute_class_weights(labels, masks)


def test_focal_loss_returns_scalar_tensor() -> None:
    # checks focal loss returns one scalar tensor
    logits = torch.tensor(
        [
            [2.0, -2.0],
            [-1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=torch.float32,
    )
    masks = torch.ones_like(targets)

    loss = focal_loss_with_mask(logits, targets, masks)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_focal_loss_is_lower_for_better_logits() -> None:
    # checks good predictions have lower loss than bad predictions
    targets = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=torch.float32,
    )
    masks = torch.ones_like(targets)

    good_logits = torch.tensor(
        [
            [5.0, -5.0],
            [-5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    bad_logits = -good_logits

    good_loss = focal_loss_with_mask(good_logits, targets, masks)
    bad_loss = focal_loss_with_mask(bad_logits, targets, masks)

    assert good_loss < bad_loss


def test_focal_loss_ignores_masked_positions() -> None:
    # checks unknown labels with mask 0 do not affect loss
    logits_a = torch.tensor([[5.0, -5.0]], dtype=torch.float32)
    logits_b = torch.tensor([[5.0, 5.0]], dtype=torch.float32)

    targets = torch.tensor([[1, 0]], dtype=torch.float32)
    masks = torch.tensor([[1, 0]], dtype=torch.float32)

    loss_a = focal_loss_with_mask(logits_a, targets, masks)
    loss_b = focal_loss_with_mask(logits_b, targets, masks)

    assert torch.isclose(loss_a, loss_b, atol=1e-6)


def test_focal_loss_accepts_class_weights() -> None:
    # checks class weights can be applied
    logits = torch.tensor(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=torch.float32,
    )
    masks = torch.ones_like(targets)
    class_weights = torch.tensor([1.0, 2.0], dtype=torch.float32)

    loss = focal_loss_with_mask(
        logits=logits,
        targets=targets,
        masks=masks,
        class_weights=class_weights,
    )

    assert torch.isfinite(loss)


def test_focal_loss_rejects_invalid_class_weights_shape() -> None:
    # checks class_weights must be 1D
    logits = torch.zeros((2, 3), dtype=torch.float32)
    targets = torch.zeros((2, 3), dtype=torch.float32)
    masks = torch.ones((2, 3), dtype=torch.float32)
    class_weights = torch.ones((1, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="class_weights must be 1D"):
        focal_loss_with_mask(logits, targets, masks, class_weights=class_weights)


def test_focal_loss_rejects_wrong_class_weights_length() -> None:
    # checks class_weights length must match number of labels
    logits = torch.zeros((2, 3), dtype=torch.float32)
    targets = torch.zeros((2, 3), dtype=torch.float32)
    masks = torch.ones((2, 3), dtype=torch.float32)
    class_weights = torch.ones((2,), dtype=torch.float32)

    with pytest.raises(ValueError, match="class_weights length must match"):
        focal_loss_with_mask(logits, targets, masks, class_weights=class_weights)


def test_focal_loss_rejects_non_positive_class_weights() -> None:
    # checks class_weights must be greater than zero
    logits = torch.zeros((2, 3), dtype=torch.float32)
    targets = torch.zeros((2, 3), dtype=torch.float32)
    masks = torch.ones((2, 3), dtype=torch.float32)
    class_weights = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="class_weights must be > 0"):
        focal_loss_with_mask(logits, targets, masks, class_weights=class_weights)


def test_focal_loss_rejects_non_2d_logits() -> None:
    # checks logits must be 2D
    logits = torch.zeros((2, 3, 4), dtype=torch.float32)
    targets = torch.zeros((2, 3, 4), dtype=torch.float32)
    masks = torch.ones((2, 3, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="logits must be 2D"):
        focal_loss_with_mask(logits, targets, masks)


def test_focal_loss_rejects_target_shape_mismatch() -> None:
    # checks targets shape must match logits shape
    logits = torch.zeros((2, 3), dtype=torch.float32)
    targets = torch.zeros((2, 2), dtype=torch.float32)
    masks = torch.ones((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="targets must match logits shape"):
        focal_loss_with_mask(logits, targets, masks)


def test_focal_loss_rejects_mask_shape_mismatch() -> None:
    # checks masks shape must match logits shape
    logits = torch.zeros((2, 3), dtype=torch.float32)
    targets = torch.zeros((2, 3), dtype=torch.float32)
    masks = torch.ones((2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="masks must match logits shape"):
        focal_loss_with_mask(logits, targets, masks)


def test_focal_loss_rejects_empty_logits() -> None:
    # checks empty logits are not allowed
    logits = torch.empty((0, 3), dtype=torch.float32)
    targets = torch.empty((0, 3), dtype=torch.float32)
    masks = torch.empty((0, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="logits must not be empty"):
        focal_loss_with_mask(logits, targets, masks)


def test_focal_loss_all_mask_zero_returns_zero_loss() -> None:
    # checks all unknown labels give zero loss and no divide by zero error
    logits = torch.tensor([[2.0, -2.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    masks = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    loss = focal_loss_with_mask(logits, targets, masks)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    #run pytest tests/ai_model/test_losses.py -q --no-cov