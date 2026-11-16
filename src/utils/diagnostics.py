"""Error-analysis statistics that explain where a segmentation model loses Dice.

Pixel level. ``PixelConfusionMatrix`` accumulates a ``num_classes x num_classes`` matrix of pixel
counts over a whole evaluation set, with rows indexed by the ground-truth class and columns by the
predicted class, using a single ``torch.bincount`` over ``target * num_classes + prediction``.
For every tumor class ``c`` it yields:
    precision_c = M[c, c] / column_sum_c         recall_c = M[c, c] / row_sum_c
    dice_c      = 2 M[c, c] / (row_sum_c + column_sum_c)
The Dice values are identical to the global aggregation of ``SegmentationMetrics``, so they can
be checked against the validation Dice stored in the checkpoint. Every false-negative pixel of a
tumor class is either predicted as background (a localization miss) or as another tumor type (a
type confusion), and the two shares are reported separately because they call for different
fixes: border and recall errors are what the loss and augmentation changes target, while type
confusions are better handled by post-processing.

Slice level. Every Figshare slice contains exactly one tumor type, so the slice's target class is
the unique non-background label of its mask, and its predicted class is the most frequent
non-background label among the predicted pixels (``BACKGROUND_LABEL`` when the model predicts no
tumor at all). ``SliceRecord`` stores the raw pixel counts of one slice so per-slice scores and
any later aggregation (by tumor type, by tumor size) can be derived without running the model
again. The localization Dice merges every tumor class into one foreground label and therefore
measures how well the tumor is outlined regardless of its predicted type.

Majority vote. ``majority_vote`` relabels every predicted tumor pixel of a slice with that
slice's predicted class. Because the ground truth never mixes tumor types inside a slice, this
post-processing cannot hurt a slice whose majority type is right, and the Dice it reaches
measures how much of the gap is caused by mixed-type predictions rather than by poor outlines.
"""

import math
from dataclasses import dataclass
from statistics import median
from typing import Dict, Final, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from ..dataset.constants import BACKGROUND_LABEL, NUM_CLASSES

UNDEFINED_SCORE: Final[float] = float("nan")
LOW_DICE_THRESHOLD: Final[float] = 0.5
SIZE_TERCILE_NAMES: Final[Tuple[str, ...]] = ("small", "medium", "large")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else UNDEFINED_SCORE


def dice_from_counts(overlap: float, predicted: float, target: float) -> float:
    return safe_ratio(2.0 * overlap, predicted + target)


class PixelConfusionMatrix:
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}.")
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.counts = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Predictions {tuple(predictions.shape)} and targets "
                f"{tuple(targets.shape)} must have the same shape."
            )
        flat_predictions = predictions.detach().flatten().long().cpu()
        flat_targets = targets.detach().flatten().long().cpu()
        if flat_predictions.numel() == 0:
            return
        highest_label = int(torch.maximum(flat_predictions.max(), flat_targets.max()))
        if highest_label >= self.num_classes or int(flat_predictions.min()) < 0:
            raise ValueError(f"Labels must lie in [0, {self.num_classes}).")
        pair_indices = flat_targets * self.num_classes + flat_predictions
        self.counts += torch.bincount(pair_indices, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )


@dataclass(frozen=True)
class ClassDiagnostics:
    class_index: int
    target_pixels: int
    predicted_pixels: int
    true_positive_pixels: int
    missed_as_background_pixels: int

    @property
    def false_negative_pixels(self) -> int:
        return self.target_pixels - self.true_positive_pixels

    @property
    def false_positive_pixels(self) -> int:
        return self.predicted_pixels - self.true_positive_pixels

    @property
    def confused_with_other_tumor_pixels(self) -> int:
        return self.false_negative_pixels - self.missed_as_background_pixels

    @property
    def precision(self) -> float:
        return safe_ratio(self.true_positive_pixels, self.predicted_pixels)

    @property
    def recall(self) -> float:
        return safe_ratio(self.true_positive_pixels, self.target_pixels)

    @property
    def dice(self) -> float:
        return dice_from_counts(
            self.true_positive_pixels, self.predicted_pixels, self.target_pixels
        )

    @property
    def iou(self) -> float:
        return safe_ratio(
            self.true_positive_pixels,
            self.predicted_pixels + self.target_pixels - self.true_positive_pixels,
        )

    @property
    def false_negative_to_false_positive_ratio(self) -> float:
        return safe_ratio(self.false_negative_pixels, self.false_positive_pixels)

    @property
    def missed_as_background_share(self) -> float:
        return safe_ratio(self.missed_as_background_pixels, self.target_pixels)

    @property
    def confused_with_other_tumor_share(self) -> float:
        return safe_ratio(self.confused_with_other_tumor_pixels, self.target_pixels)


def summarize_classes(confusion_counts: Tensor) -> Tuple[ClassDiagnostics, ...]:
    counts = confusion_counts.long()
    return tuple(
        ClassDiagnostics(
            class_index=class_index,
            target_pixels=int(counts[class_index].sum()),
            predicted_pixels=int(counts[:, class_index].sum()),
            true_positive_pixels=int(counts[class_index, class_index]),
            missed_as_background_pixels=int(counts[class_index, BACKGROUND_LABEL]),
        )
        for class_index in range(counts.shape[0])
        if class_index != BACKGROUND_LABEL
    )


def dominant_tumor_label(labels: Tensor, num_classes: int = NUM_CLASSES) -> int:
    label_counts = torch.bincount(labels.flatten().long().cpu(), minlength=num_classes)
    tumor_counts = label_counts[BACKGROUND_LABEL + 1 :]
    if int(tumor_counts.sum()) == 0:
        return BACKGROUND_LABEL
    return int(tumor_counts.argmax()) + BACKGROUND_LABEL + 1


def majority_vote(predictions: Tensor, num_classes: int = NUM_CLASSES) -> Tensor:
    voted_predictions = predictions.clone()
    for slice_index in range(predictions.shape[0]):
        slice_prediction = predictions[slice_index]
        voted_label = dominant_tumor_label(slice_prediction, num_classes)
        voted_predictions[slice_index] = torch.where(
            slice_prediction != BACKGROUND_LABEL,
            torch.full_like(slice_prediction, voted_label),
            slice_prediction,
        )
    return voted_predictions


@dataclass(frozen=True)
class SliceRecord:
    sample_name: str
    target_class: int
    predicted_class: int
    target_tumor_pixels: int
    predicted_tumor_pixels: int
    tumor_overlap_pixels: int
    predicted_target_class_pixels: int
    target_class_overlap_pixels: int

    @property
    def class_dice(self) -> float:
        return dice_from_counts(
            self.target_class_overlap_pixels,
            self.predicted_target_class_pixels,
            self.target_tumor_pixels,
        )

    @property
    def localization_dice(self) -> float:
        return dice_from_counts(
            self.tumor_overlap_pixels, self.predicted_tumor_pixels, self.target_tumor_pixels
        )

    @property
    def majority_vote_dice(self) -> float:
        if self.predicted_class == self.target_class:
            return self.localization_dice
        return (
            0.0 if self.target_tumor_pixels + self.predicted_tumor_pixels > 0 else UNDEFINED_SCORE
        )

    @property
    def has_tumor_prediction(self) -> bool:
        return self.predicted_tumor_pixels > 0

    @property
    def has_wrong_type(self) -> bool:
        return self.predicted_class not in (BACKGROUND_LABEL, self.target_class)


def build_slice_record(
    sample_name: str, prediction: Tensor, target: Tensor, num_classes: int = NUM_CLASSES
) -> SliceRecord:
    target_class = dominant_tumor_label(target, num_classes)
    predicted_tumor = prediction != BACKGROUND_LABEL
    target_tumor = target != BACKGROUND_LABEL
    predicted_target_class = prediction == target_class
    return SliceRecord(
        sample_name=sample_name,
        target_class=target_class,
        predicted_class=dominant_tumor_label(prediction, num_classes),
        target_tumor_pixels=int(target_tumor.sum()),
        predicted_tumor_pixels=int(predicted_tumor.sum()),
        tumor_overlap_pixels=int((predicted_tumor & target_tumor).sum()),
        predicted_target_class_pixels=(
            int(predicted_target_class.sum()) if target_class != BACKGROUND_LABEL else 0
        ),
        target_class_overlap_pixels=(
            int((predicted_target_class & target_tumor).sum())
            if target_class != BACKGROUND_LABEL
            else 0
        ),
    )


def build_slice_records(
    sample_names: Sequence[str],
    predictions: Tensor,
    targets: Tensor,
    num_classes: int = NUM_CLASSES,
) -> List[SliceRecord]:
    if not len(sample_names) == predictions.shape[0] == targets.shape[0]:
        raise ValueError(
            f"Got {len(sample_names)} names for {predictions.shape[0]} predictions and "
            f"{targets.shape[0]} targets."
        )
    predictions_cpu = predictions.detach().cpu()
    targets_cpu = targets.detach().cpu()
    return [
        build_slice_record(name, predictions_cpu[index], targets_cpu[index], num_classes)
        for index, name in enumerate(sample_names)
    ]


def build_slice_confusion(records: Sequence[SliceRecord], num_classes: int = NUM_CLASSES) -> Tensor:
    slice_counts = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for record in records:
        slice_counts[record.target_class, record.predicted_class] += 1
    return slice_counts[BACKGROUND_LABEL + 1 :]


def split_into_size_terciles(records: Sequence[SliceRecord]) -> Dict[str, List[SliceRecord]]:
    ordered_records = sorted(records, key=lambda record: record.target_tumor_pixels)
    tercile_count = len(SIZE_TERCILE_NAMES)
    boundaries = [
        round(index * len(ordered_records) / tercile_count) for index in range(tercile_count + 1)
    ]
    return {
        name: ordered_records[boundaries[index] : boundaries[index + 1]]
        for index, name in enumerate(SIZE_TERCILE_NAMES)
    }


def mean_of_defined(values: Sequence[float]) -> float:
    defined_values = [value for value in values if not math.isnan(value)]
    return sum(defined_values) / len(defined_values) if defined_values else UNDEFINED_SCORE


def median_of_defined(values: Sequence[float]) -> float:
    defined_values = [value for value in values if not math.isnan(value)]
    return float(median(defined_values)) if defined_values else UNDEFINED_SCORE


@dataclass(frozen=True)
class SliceGroupSummary:
    slices: int
    mean_slice_dice: float
    median_slice_dice: float
    low_dice_share: float
    no_tumor_predicted_share: float
    wrong_type_share: float
    localization_dice: float
    mean_slice_dice_by_size: Mapping[str, float]
    median_tumor_pixels_by_size: Mapping[str, float]


def summarize_slice_group(records: Sequence[SliceRecord]) -> SliceGroupSummary:
    slice_dices = [record.class_dice for record in records]
    size_terciles = split_into_size_terciles(records)
    return SliceGroupSummary(
        slices=len(records),
        mean_slice_dice=mean_of_defined(slice_dices),
        median_slice_dice=median_of_defined(slice_dices),
        low_dice_share=safe_ratio(
            sum(1 for dice in slice_dices if not math.isnan(dice) and dice < LOW_DICE_THRESHOLD),
            len(records),
        ),
        no_tumor_predicted_share=safe_ratio(
            sum(1 for record in records if not record.has_tumor_prediction), len(records)
        ),
        wrong_type_share=safe_ratio(
            sum(1 for record in records if record.has_wrong_type), len(records)
        ),
        localization_dice=dice_from_counts(
            sum(record.tumor_overlap_pixels for record in records),
            sum(record.predicted_tumor_pixels for record in records),
            sum(record.target_tumor_pixels for record in records),
        ),
        mean_slice_dice_by_size={
            name: mean_of_defined([record.class_dice for record in group])
            for name, group in size_terciles.items()
        },
        median_tumor_pixels_by_size={
            name: median_of_defined([float(record.target_tumor_pixels) for record in group])
            for name, group in size_terciles.items()
        },
    )


def summarize_slices_by_class(
    records: Sequence[SliceRecord], class_indices: Sequence[int]
) -> Dict[int, SliceGroupSummary]:
    return {
        class_index: summarize_slice_group(
            [record for record in records if record.target_class == class_index]
        )
        for class_index in class_indices
    }
