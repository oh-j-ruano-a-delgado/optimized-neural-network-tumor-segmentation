"""Tests for the per-slice whole-tumor metrics."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.utils.slice_metrics import score_slice, summarize_slices


def as_mask(rows: list[str]) -> NDArray[np.uint8]:
    return np.asarray([[int(pixel) for pixel in row] for row in rows], dtype=np.uint8)


def test_perfect_slice_scores_one() -> None:
    mask = as_mask(["0010", "0010", "0000"])
    score = score_slice(mask, mask, "baseline", "slice_1", "glioma")
    assert score.dice == pytest.approx(1.0)
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1.0)
    assert score.true_positives == 2
    assert score.detected


def test_missed_tumor_has_zero_recall_and_dice() -> None:
    truth = as_mask(["0010", "0010"])
    prediction = as_mask(["0000", "0000"])
    score = score_slice(prediction, truth, "optimized", "slice_2", "meningioma")
    assert score.dice == pytest.approx(0.0)
    assert score.recall == pytest.approx(0.0)
    assert score.precision == pytest.approx(1.0)
    assert not score.detected


def test_false_positive_slice_has_zero_precision() -> None:
    truth = as_mask(["0000", "0000"])
    prediction = as_mask(["0110", "0000"])
    score = score_slice(prediction, truth, "baseline", "slice_3", "pituitary")
    assert score.dice == pytest.approx(0.0)
    assert score.precision == pytest.approx(0.0)
    assert score.recall == pytest.approx(1.0)
    assert score.detected


def test_empty_slice_scores_one_when_prediction_is_also_empty() -> None:
    empty = as_mask(["0000", "0000"])
    score = score_slice(empty, empty, "baseline", "slice_4", "glioma")
    assert score.dice == pytest.approx(1.0)
    assert not score.detected


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="differ in shape"):
        score_slice(np.zeros((2, 2)), np.zeros((2, 3)), "baseline", "slice", "glioma")


def test_partial_overlap_scores_expected_values() -> None:
    truth = as_mask(["1111", "1100"])
    prediction = as_mask(["1100", "1110"])
    score = score_slice(prediction, truth, "baseline", "slice_5", "glioma")
    assert score.true_positives == 4
    assert score.dice == pytest.approx(8 / 11)
    assert score.precision == pytest.approx(4 / 5)
    assert score.recall == pytest.approx(4 / 6)


def test_summary_reports_detection_by_class_and_pooled_rates() -> None:
    tumor = as_mask(["0110", "0110"])
    empty = as_mask(["0000", "0000"])
    scores = [
        score_slice(tumor, tumor, "baseline", "a", "glioma"),
        score_slice(empty, tumor, "baseline", "b", "glioma"),
        score_slice(tumor, tumor, "baseline", "c", "pituitary"),
        score_slice(tumor, empty, "baseline", "d", "pituitary"),
    ]
    summary = summarize_slices(scores)

    assert summary.slices == 4
    assert summary.tumor_slices == 3
    assert summary.empty_ground_truth_slices == 1
    assert summary.predicted_empty_slices == 1
    assert summary.missed_tumor_slices == 1
    assert summary.false_positive_slices == 1
    assert summary.detection_rate == pytest.approx(2 / 3)
    assert summary.detection_rate_by_class == {"glioma": 0.5, "pituitary": 1.0}
    assert summary.precision == pytest.approx(8 / 12)
    assert summary.recall == pytest.approx(8 / 12)
    assert summary.dice_mean == pytest.approx(2 / 3)
    assert summary.dice_median == pytest.approx(1.0)


def test_summary_serializes_to_rounded_dict() -> None:
    tumor = as_mask(["0110", "0110"])
    payload = summarize_slices(
        [score_slice(tumor, tumor, "baseline", "a", "glioma")]
    ).to_dict()
    assert payload["detection_rate"] == 1.0
    assert payload["dice"]["mean"] == 1.0
    assert payload["detection_rate_by_class"] == {"glioma": 1.0}


def test_empty_score_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty list"):
        summarize_slices([])
