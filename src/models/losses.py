"""Segmentation losses for the severely imbalanced background/tumor problem.

Every region loss works on softmax probabilities ``p`` and one-hot targets ``g``, sums over the
whole batch and all pixels (batch-level aggregation), excludes the background channel by default
and averages only over the classes present in the batch. Each Figshare slice contains a single
tumor type, so classes missing from the batch carry no overlap information and would otherwise
count as perfect scores and dilute the loss.

DiceLoss:          1 - (2 * sum(p g) + eps) / (sum(p) + sum(g) + eps)
FocalLoss:         mean over pixels of -alpha_t * (1 - p_t)^gamma * log(p_t); with gamma = 0 and
                   no alpha it is exactly the multi-class cross-entropy.
FocalTverskyLoss:  for each present class c,
                       TP_c = sum(p g), FN_c = sum((1 - p) g), FP_c = sum(p (1 - g))
                       TI_c = (TP_c + eps) / (TP_c + alpha FN_c + beta FP_c + eps)
                       loss = mean_c (1 - TI_c)^gamma
                   (Abraham & Khan, 2019). alpha > beta penalizes missed tumor pixels more than
                   false alarms; the defaults alpha = 0.7, beta = 0.3 target the under-segmented,
                   diffuse glioma borders. 1 - TI_c is computed directly as
                   (alpha FN_c + beta FP_c) / (TP_c + alpha FN_c + beta FP_c + eps), which is
                   non-negative by construction, and it is floored at a tiny epsilon so gamma < 1
                   (the paper's own exponent 1 / (4/3) = 0.75) never produces an infinite gradient
                   at a perfect prediction. With gamma = 4/3 > 1 the class with the lowest Tversky
                   index receives the largest gradient (glioma at TI ~ 0.6 gets ~1.5x the gradient
                   of meningioma at TI ~ 0.89), whereas gamma < 1 shifts the emphasis to the
                   easier classes. With alpha = beta = 0.5 and gamma = 1 the loss equals DiceLoss
                   up to the smoothing constant.

Penalty-method combinations, ``Loss = region_weight * region term + pixel_weight * pixel term``:
    "dice_focal"           Dice + Focal (Phase 2-4 default).
    "focal_tversky_focal"  Focal Tversky + Focal.
    "dice_ce"              Dice + cross-entropy, built as DiceFocalLoss with gamma = 0, the
                           nnU-Net default and the control run for the Focal Tversky loss.
``build_criterion`` creates any of them from a ``LossSettings`` value.
"""

from dataclasses import dataclass
from typing import Final, Literal, Optional, Sequence, Tuple, get_args

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils.tensor_ops import one_hot_encode
from .unet import DEFAULT_NUM_CLASSES

LossName = Literal["dice_focal", "focal_tversky_focal", "dice_ce"]

DEFAULT_DICE_WEIGHT: Final[float] = 0.5
DEFAULT_FOCAL_WEIGHT: Final[float] = 0.5
DEFAULT_FOCAL_GAMMA: Final[float] = 2.0
DEFAULT_SMOOTHING_FACTOR: Final[float] = 1e-5
DEFAULT_TVERSKY_ALPHA: Final[float] = 0.7
DEFAULT_TVERSKY_BETA: Final[float] = 0.3
DEFAULT_TVERSKY_GAMMA: Final[float] = 4.0 / 3.0
DEFAULT_LOSS_NAME: Final[LossName] = "dice_focal"
LOSS_NAMES: Final[Tuple[str, ...]] = get_args(LossName)
CROSS_ENTROPY_GAMMA: Final[float] = 0.0
MINIMUM_TVERSKY_DISTANCE: Final[float] = 1e-7
SUMMED_DIMENSIONS: Final[Tuple[int, int, int]] = (0, 2, 3)


def validate_smoothing_factor(smoothing_factor: float) -> None:
    if smoothing_factor <= 0:
        raise ValueError(f"smoothing_factor must be positive, got {smoothing_factor}.")


def validate_combination_weights(region_weight: float, pixel_weight: float) -> None:
    if region_weight < 0 or pixel_weight < 0:
        raise ValueError(
            f"Loss weights must be non-negative, got region_weight={region_weight}, "
            f"pixel_weight={pixel_weight}."
        )
    if region_weight + pixel_weight == 0:
        raise ValueError("At least one loss weight must be positive.")


def class_probability_maps(
    logits: Tensor, target: Tensor, num_classes: int, include_background: bool
) -> Tuple[Tensor, Tensor]:
    probabilities = torch.softmax(logits, dim=1)
    targets = one_hot_encode(target, num_classes)
    if include_background:
        return probabilities, targets
    return probabilities[:, 1:], targets[:, 1:]


def present_class_mask(targets: Tensor) -> Tensor:
    return targets.sum(dim=SUMMED_DIMENSIONS) > 0


def zero_loss_like(logits: Tensor) -> Tensor:
    return logits.sum() * 0.0


class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        validate_smoothing_factor(smoothing_factor)
        self.num_classes = num_classes
        self.smoothing_factor = smoothing_factor
        self.include_background = include_background

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probabilities, targets = class_probability_maps(
            logits, target, self.num_classes, self.include_background
        )
        intersections = (probabilities * targets).sum(dim=SUMMED_DIMENSIONS)
        cardinalities = probabilities.sum(dim=SUMMED_DIMENSIONS) + targets.sum(
            dim=SUMMED_DIMENSIONS
        )
        dice_scores = (2.0 * intersections + self.smoothing_factor) / (
            cardinalities + self.smoothing_factor
        )
        present_classes = present_class_mask(targets)
        if not present_classes.any():
            return zero_loss_like(logits)
        return 1.0 - dice_scores[present_classes].mean()


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = DEFAULT_TVERSKY_ALPHA,
        beta: float = DEFAULT_TVERSKY_BETA,
        gamma: float = DEFAULT_TVERSKY_GAMMA,
        num_classes: int = DEFAULT_NUM_CLASSES,
        smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        if alpha < 0 or beta < 0 or alpha + beta == 0:
            raise ValueError(
                f"alpha and beta must be non-negative and not both zero, got alpha={alpha}, "
                f"beta={beta}."
            )
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}.")
        validate_smoothing_factor(smoothing_factor)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.smoothing_factor = smoothing_factor
        self.include_background = include_background

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probabilities, targets = class_probability_maps(
            logits, target, self.num_classes, self.include_background
        )
        true_positives = (probabilities * targets).sum(dim=SUMMED_DIMENSIONS)
        false_negatives = ((1.0 - probabilities) * targets).sum(dim=SUMMED_DIMENSIONS)
        false_positives = (probabilities * (1.0 - targets)).sum(dim=SUMMED_DIMENSIONS)
        weighted_errors = self.alpha * false_negatives + self.beta * false_positives
        tversky_distances = weighted_errors / (
            true_positives + weighted_errors + self.smoothing_factor
        )
        focal_tversky_losses = tversky_distances.clamp_min(MINIMUM_TVERSKY_DISTANCE).pow(
            self.gamma
        )
        present_classes = present_class_mask(targets)
        if not present_classes.any():
            return zero_loss_like(logits)
        return focal_tversky_losses[present_classes].mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = DEFAULT_FOCAL_GAMMA,
        alpha: Optional[Sequence[float]] = None,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}.")
        if alpha is not None and len(alpha) != num_classes:
            raise ValueError(
                f"alpha must provide one weight per class ({num_classes} weights), got {len(alpha)}."
            )
        self.gamma = gamma
        self.num_classes = num_classes
        self.register_buffer(
            "alpha",
            None if alpha is None else torch.tensor(alpha, dtype=torch.float32),
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        target_log_probabilities = log_probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
        target_probabilities = target_log_probabilities.exp()
        focal_weights = (1.0 - target_probabilities).pow(self.gamma)
        losses = focal_weights * (-target_log_probabilities)
        if self.alpha is not None:
            losses = losses * self.alpha.to(losses.dtype)[target]
        return losses.mean()


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = DEFAULT_DICE_WEIGHT,
        focal_weight: float = DEFAULT_FOCAL_WEIGHT,
        num_classes: int = DEFAULT_NUM_CLASSES,
        gamma: float = DEFAULT_FOCAL_GAMMA,
        alpha: Optional[Sequence[float]] = None,
        include_background: bool = False,
        smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
    ) -> None:
        super().__init__()
        validate_combination_weights(dice_weight, focal_weight)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(num_classes, smoothing_factor, include_background)
        self.focal_loss = FocalLoss(gamma, alpha, num_classes)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.dice_weight * self.dice_loss(logits, target) + self.focal_weight * self.focal_loss(
            logits, target
        )


class FocalTverskyFocalLoss(nn.Module):
    def __init__(
        self,
        tversky_weight: float = DEFAULT_DICE_WEIGHT,
        focal_weight: float = DEFAULT_FOCAL_WEIGHT,
        num_classes: int = DEFAULT_NUM_CLASSES,
        tversky_alpha: float = DEFAULT_TVERSKY_ALPHA,
        tversky_beta: float = DEFAULT_TVERSKY_BETA,
        tversky_gamma: float = DEFAULT_TVERSKY_GAMMA,
        focal_gamma: float = DEFAULT_FOCAL_GAMMA,
        focal_alpha: Optional[Sequence[float]] = None,
        include_background: bool = False,
        smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
    ) -> None:
        super().__init__()
        validate_combination_weights(tversky_weight, focal_weight)
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        self.focal_tversky_loss = FocalTverskyLoss(
            tversky_alpha,
            tversky_beta,
            tversky_gamma,
            num_classes,
            smoothing_factor,
            include_background,
        )
        self.focal_loss = FocalLoss(focal_gamma, focal_alpha, num_classes)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.tversky_weight * self.focal_tversky_loss(
            logits, target
        ) + self.focal_weight * self.focal_loss(logits, target)


@dataclass(frozen=True)
class LossSettings:
    name: LossName = DEFAULT_LOSS_NAME
    region_weight: float = DEFAULT_DICE_WEIGHT
    pixel_weight: float = DEFAULT_FOCAL_WEIGHT
    focal_gamma: float = DEFAULT_FOCAL_GAMMA
    tversky_alpha: float = DEFAULT_TVERSKY_ALPHA
    tversky_beta: float = DEFAULT_TVERSKY_BETA
    tversky_gamma: float = DEFAULT_TVERSKY_GAMMA

    def __post_init__(self) -> None:
        if self.name not in LOSS_NAMES:
            raise ValueError(f"Unknown loss '{self.name}', expected one of {LOSS_NAMES}.")


def build_criterion(
    settings: LossSettings = LossSettings(), num_classes: int = DEFAULT_NUM_CLASSES
) -> nn.Module:
    if settings.name == "focal_tversky_focal":
        return FocalTverskyFocalLoss(
            tversky_weight=settings.region_weight,
            focal_weight=settings.pixel_weight,
            num_classes=num_classes,
            tversky_alpha=settings.tversky_alpha,
            tversky_beta=settings.tversky_beta,
            tversky_gamma=settings.tversky_gamma,
            focal_gamma=settings.focal_gamma,
        )
    pixel_gamma = CROSS_ENTROPY_GAMMA if settings.name == "dice_ce" else settings.focal_gamma
    return DiceFocalLoss(
        dice_weight=settings.region_weight,
        focal_weight=settings.pixel_weight,
        num_classes=num_classes,
        gamma=pixel_gamma,
    )
