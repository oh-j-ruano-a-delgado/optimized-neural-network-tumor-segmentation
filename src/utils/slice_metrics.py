"""Per-slice whole-tumor metrics for the held-out test set.

Global (pixel-pooled) Dice rewards large tumors and hides whether the model finds small
lesions at all. These helpers score each 2D slice independently so detection rate, missed
tumors and empty predictions can be reported alongside the pooled scores.

Conventions: Dice, precision and recall are 1.0 when their denominator is zero (nothing to
find or nothing predicted), matching the usual ``zero_division=1`` behaviour.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SliceScore:
    model: str
    sample_name: str
    class_name: str
    dice: float
    precision: float
    recall: float
    tumor_pixels: int
    predicted_pixels: int
    true_positives: int

    @property
    def detected(self) -> bool:
        return self.predicted_pixels > 0


@dataclass(frozen=True)
class SliceSummary:
    slices: int
    tumor_slices: int
    empty_ground_truth_slices: int
    predicted_empty_slices: int
    missed_tumor_slices: int
    false_positive_slices: int
    detection_rate: float
    precision: float
    recall: float
    dice_mean: float
    dice_median: float
    dice_p05: float
    dice_p95: float
    detection_rate_by_class: Mapping[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "slices": self.slices,
            "tumor_slices": self.tumor_slices,
            "empty_ground_truth_slices": self.empty_ground_truth_slices,
            "predicted_empty_slices": self.predicted_empty_slices,
            "missed_tumor_slices": self.missed_tumor_slices,
            "false_positive_slices": self.false_positive_slices,
            "detection_rate": round(self.detection_rate, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "dice": {
                "mean": round(self.dice_mean, 6),
                "median": round(self.dice_median, 6),
                "p05": round(self.dice_p05, 6),
                "p95": round(self.dice_p95, 6),
            },
            "detection_rate_by_class": {
                name: round(rate, 6) for name, rate in self.detection_rate_by_class.items()
            },
        }


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 1.0


def score_slice(
    prediction: NDArray[np.generic],
    ground_truth: NDArray[np.generic],
    model: str,
    sample_name: str,
    class_name: str,
) -> SliceScore:
    predicted = np.asarray(prediction) != 0
    truth = np.asarray(ground_truth) != 0
    if predicted.shape != truth.shape:
        raise ValueError(
            f"Prediction and ground truth differ in shape "
            f"({predicted.shape} vs {truth.shape}) for '{sample_name}'."
        )
    true_positives = int(np.logical_and(predicted, truth).sum())
    predicted_pixels = int(predicted.sum())
    tumor_pixels = int(truth.sum())
    false_positives = predicted_pixels - true_positives
    false_negatives = tumor_pixels - true_positives
    dice_denominator = 2 * true_positives + false_positives + false_negatives
    return SliceScore(
        model=model,
        sample_name=sample_name,
        class_name=class_name,
        dice=safe_ratio(2 * true_positives, dice_denominator),
        precision=safe_ratio(true_positives, true_positives + false_positives),
        recall=safe_ratio(true_positives, true_positives + false_negatives),
        tumor_pixels=tumor_pixels,
        predicted_pixels=predicted_pixels,
        true_positives=true_positives,
    )


def summarize_slices(scores: Sequence[SliceScore]) -> SliceSummary:
    if not scores:
        raise ValueError("Cannot summarize an empty list of slice scores.")
    tumor_scores = [score for score in scores if score.tumor_pixels > 0]
    empty_scores = [score for score in scores if score.tumor_pixels == 0]
    detected = [score for score in tumor_scores if score.detected]
    dice_values = np.asarray([score.dice for score in tumor_scores], dtype=np.float64)
    class_detection: Dict[str, float] = {}
    for class_name in sorted({score.class_name for score in tumor_scores}):
        class_scores = [score for score in tumor_scores if score.class_name == class_name]
        class_detection[class_name] = safe_ratio(
            sum(score.detected for score in class_scores), len(class_scores)
        )
    true_positives = sum(score.true_positives for score in scores)
    predicted_pixels = sum(score.predicted_pixels for score in scores)
    tumor_pixels = sum(score.tumor_pixels for score in scores)
    return SliceSummary(
        slices=len(scores),
        tumor_slices=len(tumor_scores),
        empty_ground_truth_slices=len(empty_scores),
        predicted_empty_slices=sum(not score.detected for score in scores),
        missed_tumor_slices=len(tumor_scores) - len(detected),
        false_positive_slices=sum(score.detected for score in empty_scores),
        detection_rate=safe_ratio(len(detected), len(tumor_scores)),
        precision=safe_ratio(true_positives, predicted_pixels),
        recall=safe_ratio(true_positives, tumor_pixels),
        dice_mean=float(dice_values.mean()),
        dice_median=float(np.median(dice_values)),
        dice_p05=float(np.percentile(dice_values, 5)),
        dice_p95=float(np.percentile(dice_values, 95)),
        detection_rate_by_class=class_detection,
    )
