"""Tests for the Dice, Focal, Focal Tversky and combined losses in ``src.models.losses``.

Targets are tiny 4x4 slices holding a single tumor class, mirroring the Figshare masks. Classes
without ground-truth pixels in the batch must not contribute to the region losses, so a prediction
that only matches the present class must score close to zero even though the other tumor channels
carry no overlap. The Focal Tversky checks cover its two defining properties: it reduces to the
Dice loss for alpha = beta = 0.5 and gamma = 1, and alpha > beta penalizes missed tumor pixels more
than the same number of false-positive pixels.
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.losses import (
    DiceFocalLoss,
    DiceLoss,
    FocalLoss,
    FocalTverskyFocalLoss,
    FocalTverskyLoss,
    LossSettings,
    build_criterion,
)
from src.utils.tensor_ops import one_hot_encode

NUM_CLASSES = 4


def build_target_with_single_class(class_index: int = 1) -> torch.Tensor:
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, 0, :2] = class_index
    target[0, 1, :2] = class_index
    return target


def logits_perfectly_matching(target: torch.Tensor) -> torch.Tensor:
    return one_hot_encode(target, NUM_CLASSES) * 20.0


def test_dice_loss_near_zero_on_perfect_prediction() -> None:
    target = build_target_with_single_class()
    loss = DiceLoss(num_classes=NUM_CLASSES)(logits_perfectly_matching(target), target)
    assert loss.item() < 0.01


def test_dice_loss_near_one_when_class_is_missed() -> None:
    target = build_target_with_single_class()
    wrong_logits = torch.zeros(1, NUM_CLASSES, 4, 4)
    wrong_logits[:, 2] = 10.0
    loss = DiceLoss(num_classes=NUM_CLASSES)(wrong_logits, target)
    assert loss.item() > 0.95


def test_dice_loss_ignores_background_by_default() -> None:
    target = build_target_with_single_class()
    wrong_logits = torch.zeros(1, NUM_CLASSES, 4, 4)
    wrong_logits[:, 2] = 10.0
    without_background = DiceLoss(num_classes=NUM_CLASSES)(wrong_logits, target)
    with_background = DiceLoss(num_classes=NUM_CLASSES, include_background=True)(
        wrong_logits, target
    )
    assert with_background.item() > without_background.item()


def test_dice_loss_class_absent_from_batch_is_excluded() -> None:
    target = build_target_with_single_class(class_index=1)
    logits = logits_perfectly_matching(target)
    loss = DiceLoss(num_classes=NUM_CLASSES)(logits, target)
    assert loss.item() < 0.01


def test_focal_loss_reduces_to_cross_entropy_when_gamma_zero() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, NUM_CLASSES, 4, 4)
    target = torch.randint(0, NUM_CLASSES, (2, 4, 4))
    focal = FocalLoss(gamma=0.0, num_classes=NUM_CLASSES)(logits, target)
    cross_entropy = F.cross_entropy(logits, target)
    assert torch.allclose(focal, cross_entropy)


def test_focal_loss_downweights_confident_examples() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, NUM_CLASSES, 4, 4)
    target = torch.randint(0, NUM_CLASSES, (2, 4, 4))
    cross_entropy = F.cross_entropy(logits, target)
    focal = FocalLoss(gamma=2.0, num_classes=NUM_CLASSES)(logits, target)
    assert focal.item() <= cross_entropy.item()


def test_focal_loss_zero_alpha_weights_vanish() -> None:
    target = build_target_with_single_class()
    logits = logits_perfectly_matching(target)
    loss = FocalLoss(alpha=[0.0] * NUM_CLASSES, num_classes=NUM_CLASSES)(logits, target)
    assert loss.item() == pytest.approx(0.0)


def test_focal_loss_rejects_mismatched_alpha_length() -> None:
    with pytest.raises(ValueError):
        FocalLoss(alpha=[1.0, 1.0], num_classes=NUM_CLASSES)


def test_combined_loss_is_weighted_sum_of_components() -> None:
    target = build_target_with_single_class()
    logits = torch.randn(1, NUM_CLASSES, 4, 4)
    combined = DiceFocalLoss(
        dice_weight=0.3, focal_weight=0.7, num_classes=NUM_CLASSES
    )(logits, target)
    manual = (
        0.3 * DiceLoss(num_classes=NUM_CLASSES)(logits, target)
        + 0.7 * FocalLoss(num_classes=NUM_CLASSES)(logits, target)
    )
    assert torch.allclose(combined, manual)


def test_combined_loss_rejects_zero_total_weight() -> None:
    with pytest.raises(ValueError):
        DiceFocalLoss(dice_weight=0.0, focal_weight=0.0)


def test_combined_loss_backpropagates_finite_gradients() -> None:
    target = build_target_with_single_class()
    logits = torch.randn(1, NUM_CLASSES, 4, 4, requires_grad=True)
    loss = DiceFocalLoss(num_classes=NUM_CLASSES)(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_tversky_reduces_to_dice_with_symmetric_weights() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, NUM_CLASSES, 8, 8)
    target = torch.randint(0, NUM_CLASSES, (2, 8, 8))
    tversky = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.0, num_classes=NUM_CLASSES)
    dice = DiceLoss(num_classes=NUM_CLASSES)
    assert torch.allclose(tversky(logits, target), dice(logits, target), atol=1e-5)


def test_focal_tversky_near_zero_on_perfect_prediction() -> None:
    target = build_target_with_single_class()
    loss = FocalTverskyLoss(num_classes=NUM_CLASSES)(logits_perfectly_matching(target), target)
    assert loss.item() < 0.01


def test_focal_tversky_near_one_when_class_is_missed() -> None:
    target = build_target_with_single_class()
    wrong_logits = torch.zeros(1, NUM_CLASSES, 4, 4)
    wrong_logits[:, 2] = 20.0
    loss = FocalTverskyLoss(num_classes=NUM_CLASSES)(wrong_logits, target)
    assert loss.item() > 0.95


def test_focal_tversky_penalizes_false_negatives_more_than_false_positives() -> None:
    target = build_target_with_single_class()
    under_segmented = logits_perfectly_matching(target)
    under_segmented[0, :, 0, 0] = 0.0
    under_segmented[0, 0, 0, 0] = 20.0
    over_segmented = logits_perfectly_matching(target)
    over_segmented[0, :, 3, 3] = 0.0
    over_segmented[0, 1, 3, 3] = 20.0
    loss = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.0, num_classes=NUM_CLASSES)
    assert loss(under_segmented, target).item() > loss(over_segmented, target).item()


def test_focal_tversky_class_absent_from_batch_is_excluded() -> None:
    target = build_target_with_single_class(class_index=3)
    loss = FocalTverskyLoss(num_classes=NUM_CLASSES)(logits_perfectly_matching(target), target)
    assert loss.item() < 0.01


def test_focal_tversky_returns_zero_without_foreground() -> None:
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    logits = torch.randn(1, NUM_CLASSES, 4, 4, requires_grad=True)
    loss = FocalTverskyLoss(num_classes=NUM_CLASSES)(logits, target)
    loss.backward()
    assert loss.item() == pytest.approx(0.0)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("gamma", [0.75, 4.0 / 3.0])
def test_focal_tversky_gradients_stay_finite_on_saturated_logits(gamma: float) -> None:
    target = build_target_with_single_class()
    logits = (one_hot_encode(target, NUM_CLASSES) * 200.0).requires_grad_(True)
    loss = FocalTverskyLoss(gamma=gamma, num_classes=NUM_CLASSES)(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_focal_tversky_higher_gamma_emphasizes_harder_class() -> None:
    easy_distance = torch.tensor(0.11)
    hard_distance = torch.tensor(0.4)
    gamma = 4.0 / 3.0
    easy_gradient = gamma * easy_distance.pow(gamma - 1.0)
    hard_gradient = gamma * hard_distance.pow(gamma - 1.0)
    assert (hard_gradient / easy_gradient).item() == pytest.approx(1.53, abs=0.01)


@pytest.mark.parametrize(
    "parameters",
    [{"alpha": -0.1}, {"alpha": 0.0, "beta": 0.0}, {"gamma": 0.0}, {"smoothing_factor": 0.0}],
)
def test_focal_tversky_rejects_invalid_parameters(parameters: dict) -> None:
    with pytest.raises(ValueError):
        FocalTverskyLoss(num_classes=NUM_CLASSES, **parameters)


def test_focal_tversky_focal_is_weighted_sum_of_components() -> None:
    torch.manual_seed(1)
    target = build_target_with_single_class()
    logits = torch.randn(1, NUM_CLASSES, 4, 4)
    combined = FocalTverskyFocalLoss(
        tversky_weight=0.4, focal_weight=0.6, num_classes=NUM_CLASSES
    )(logits, target)
    manual = 0.4 * FocalTverskyLoss(num_classes=NUM_CLASSES)(
        logits, target
    ) + 0.6 * FocalLoss(num_classes=NUM_CLASSES)(logits, target)
    assert torch.allclose(combined, manual)


def test_focal_tversky_focal_backpropagates_finite_gradients() -> None:
    target = build_target_with_single_class()
    logits = torch.randn(1, NUM_CLASSES, 4, 4, requires_grad=True)
    FocalTverskyFocalLoss(num_classes=NUM_CLASSES)(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_build_criterion_creates_each_loss() -> None:
    assert isinstance(build_criterion(LossSettings("dice_focal"), NUM_CLASSES), DiceFocalLoss)
    assert isinstance(
        build_criterion(LossSettings("focal_tversky_focal"), NUM_CLASSES), FocalTverskyFocalLoss
    )
    dice_ce = build_criterion(LossSettings("dice_ce"), NUM_CLASSES)
    assert isinstance(dice_ce, DiceFocalLoss)
    assert dice_ce.focal_loss.gamma == 0.0


def test_build_criterion_dice_ce_is_dice_plus_cross_entropy() -> None:
    torch.manual_seed(2)
    logits = torch.randn(2, NUM_CLASSES, 4, 4)
    target = torch.randint(0, NUM_CLASSES, (2, 4, 4))
    criterion = build_criterion(LossSettings("dice_ce", region_weight=0.5, pixel_weight=0.5))
    expected = 0.5 * DiceLoss(num_classes=NUM_CLASSES)(logits, target) + 0.5 * F.cross_entropy(
        logits, target
    )
    assert torch.allclose(criterion(logits, target), expected)


def test_build_criterion_forwards_tversky_settings() -> None:
    criterion = build_criterion(
        LossSettings("focal_tversky_focal", tversky_alpha=0.3, tversky_beta=0.7, tversky_gamma=0.75)
    )
    assert criterion.focal_tversky_loss.alpha == 0.3
    assert criterion.focal_tversky_loss.beta == 0.7
    assert criterion.focal_tversky_loss.gamma == 0.75


def test_loss_settings_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown loss"):
        LossSettings("hinge")
