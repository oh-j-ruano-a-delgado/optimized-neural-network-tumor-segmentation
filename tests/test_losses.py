import pytest
import torch
import torch.nn.functional as F

from src.models.losses import DiceFocalLoss, DiceLoss, FocalLoss
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
    # Class 3 gets a confident prediction but has no ground truth pixels, so it
    # must not contribute: only class 1 is present in the batch.
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
