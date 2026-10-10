from typing import Final, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils.tensor_ops import one_hot_encode
from .unet import DEFAULT_NUM_CLASSES

DEFAULT_DICE_WEIGHT: Final[float] = 0.5
DEFAULT_FOCAL_WEIGHT: Final[float] = 0.5
DEFAULT_FOCAL_GAMMA: Final[float] = 2.0
DEFAULT_SMOOTHING_FACTOR: Final[float] = 1e-5


class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        if smoothing_factor <= 0:
            raise ValueError(f"smoothing_factor must be positive, got {smoothing_factor}.")
        self.num_classes = num_classes
        self.smoothing_factor = smoothing_factor
        self.include_background = include_background

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probabilities = torch.softmax(logits, dim=1)
        targets = one_hot_encode(target, self.num_classes)
        if not self.include_background:
            probabilities = probabilities[:, 1:]
            targets = targets[:, 1:]

        intersections = (probabilities * targets).sum(dim=(0, 2, 3))
        cardinalities = probabilities.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        dice_scores = (2.0 * intersections + self.smoothing_factor) / (
            cardinalities + self.smoothing_factor
        )

        # Each slice contains a single tumor type, so classes missing from the
        # batch carry no overlap information and would otherwise count as
        # perfect dice and dilute the loss.
        present_classes = targets.sum(dim=(0, 2, 3)) > 0
        if not present_classes.any():
            return logits.sum() * 0.0
        return 1.0 - dice_scores[present_classes].mean()


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
    """Penalty method combining both losses: `Loss = α · Dice + β · Focal`."""

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
        if dice_weight < 0 or focal_weight < 0:
            raise ValueError(
                f"Loss weights must be non-negative, got dice_weight={dice_weight}, "
                f"focal_weight={focal_weight}."
            )
        if dice_weight + focal_weight == 0:
            raise ValueError("At least one loss weight must be positive.")
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(num_classes, smoothing_factor, include_background)
        self.focal_loss = FocalLoss(gamma, alpha, num_classes)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.dice_weight * self.dice_loss(logits, target) + self.focal_weight * self.focal_loss(
            logits, target
        )
