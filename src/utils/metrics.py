from dataclasses import dataclass
from typing import Final, List, Tuple

import torch
from torch import Tensor

from ..dataset.constants import NUM_CLASSES
from .tensor_ops import one_hot_encode

DEFAULT_NUM_CLASSES: Final[int] = NUM_CLASSES
ABSENT_CLASS_SCORE: Final[float] = float("nan")


@dataclass(frozen=True)
class SegmentationReport:
    class_indices: Tuple[int, ...]
    dice_by_class: Tensor
    iou_by_class: Tensor

    @property
    def mean_dice(self) -> float:
        return self.dice_by_class.nanmean().item()

    @property
    def mean_iou(self) -> float:
        return self.iou_by_class.nanmean().item()


class SegmentationMetrics:
    """Accumulates pixel intersections, predictions and targets per class over a
    whole evaluation set, so the final Dice/IoU is computed globally instead of
    being averaged across batches."""

    def __init__(
        self, num_classes: int = DEFAULT_NUM_CLASSES, include_background: bool = False
    ) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}.")
        self.num_classes = num_classes
        self.include_background = include_background
        self.reset()

    @property
    def class_indices(self) -> List[int]:
        return list(range(self.num_classes)) if self.include_background else list(
            range(1, self.num_classes)
        )

    def reset(self) -> None:
        self._intersections = torch.zeros(self.num_classes, dtype=torch.float64)
        self._predicted_totals = torch.zeros(self.num_classes, dtype=torch.float64)
        self._target_totals = torch.zeros(self.num_classes, dtype=torch.float64)

    def update(self, logits: Tensor, target: Tensor) -> None:
        predictions = (logits.argmax(dim=1) if logits.ndim == 4 else logits).cpu()
        targets = target.cpu()
        predicted_one_hot = one_hot_encode(predictions, self.num_classes)
        target_one_hot = one_hot_encode(targets, self.num_classes)
        self._intersections += (predicted_one_hot * target_one_hot).sum(dim=(0, 2, 3))
        self._predicted_totals += predicted_one_hot.sum(dim=(0, 2, 3))
        self._target_totals += target_one_hot.sum(dim=(0, 2, 3))

    def compute(self) -> SegmentationReport:
        intersections = self._intersections
        dice_scores = 2.0 * intersections / (self._predicted_totals + self._target_totals)
        union_totals = self._predicted_totals + self._target_totals - intersections
        iou_scores = intersections / union_totals
        # A class absent from both the predictions and the ground truth has an
        # undefined score, so it is marked as NaN and ignored by the nanmean.
        absent_classes = (self._predicted_totals + self._target_totals) == 0
        dice_scores = dice_scores.masked_fill(absent_classes, ABSENT_CLASS_SCORE)
        iou_scores = iou_scores.masked_fill(absent_classes, ABSENT_CLASS_SCORE)

        indices = torch.tensor(self.class_indices)
        return SegmentationReport(
            class_indices=tuple(self.class_indices),
            dice_by_class=dice_scores[indices],
            iou_by_class=iou_scores[indices],
        )
