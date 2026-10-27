"""Tests for the Step 0 error-analysis statistics in ``src.utils.diagnostics``.

The fixtures are tiny hand-built 4x4 slices whose pixel counts can be verified by eye: rows of the
confusion matrix are ground-truth classes and columns are predicted classes, and every slice holds
a single tumor type, mirroring the Figshare masks.
"""

import math

import pytest
import torch

from src.utils.diagnostics import (
    PixelConfusionMatrix,
    build_slice_confusion,
    build_slice_record,
    build_slice_records,
    dominant_tumor_label,
    majority_vote,
    split_into_size_terciles,
    summarize_classes,
    summarize_slice_group,
)
from src.utils.metrics import SegmentationMetrics

NUM_CLASSES = 4


def build_glioma_slice() -> torch.Tensor:
    target = torch.zeros(4, 4, dtype=torch.long)
    target[:2, :2] = 2
    return target


def build_mixed_prediction() -> torch.Tensor:
    prediction = torch.zeros(4, 4, dtype=torch.long)
    prediction[0, :2] = 2
    prediction[1, 0] = 1
    prediction[3, 3] = 2
    return prediction


def test_confusion_matrix_counts_pixel_pairs() -> None:
    matrix = PixelConfusionMatrix(NUM_CLASSES)
    matrix.update(build_mixed_prediction().unsqueeze(0), build_glioma_slice().unsqueeze(0))
    expected = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    expected[0, 0] = 11
    expected[0, 2] = 1
    expected[2, 2] = 2
    expected[2, 1] = 1
    expected[2, 0] = 1
    assert torch.equal(matrix.counts, expected)
    assert int(matrix.counts.sum()) == 16


def test_confusion_matrix_rejects_out_of_range_labels() -> None:
    matrix = PixelConfusionMatrix(NUM_CLASSES)
    with pytest.raises(ValueError):
        matrix.update(torch.full((1, 2, 2), 4), torch.zeros(1, 2, 2, dtype=torch.long))


def test_class_summary_splits_false_negatives_by_cause() -> None:
    matrix = PixelConfusionMatrix(NUM_CLASSES)
    matrix.update(build_mixed_prediction().unsqueeze(0), build_glioma_slice().unsqueeze(0))
    glioma = next(item for item in summarize_classes(matrix.counts) if item.class_index == 2)
    assert glioma.target_pixels == 4
    assert glioma.predicted_pixels == 3
    assert glioma.false_negative_pixels == 2
    assert glioma.false_positive_pixels == 1
    assert glioma.missed_as_background_pixels == 1
    assert glioma.confused_with_other_tumor_pixels == 1
    assert glioma.precision == pytest.approx(2 / 3)
    assert glioma.recall == pytest.approx(0.5)
    assert glioma.dice == pytest.approx(4 / 7)
    assert glioma.false_negative_to_false_positive_ratio == pytest.approx(2.0)


def test_class_summary_dice_matches_segmentation_metrics() -> None:
    torch.manual_seed(0)
    predictions = torch.randint(0, NUM_CLASSES, (3, 8, 8))
    targets = torch.randint(0, NUM_CLASSES, (3, 8, 8))
    matrix = PixelConfusionMatrix(NUM_CLASSES)
    matrix.update(predictions, targets)
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)
    metrics.update(predictions, targets)
    expected_dice = metrics.compute().dice_by_class.tolist()
    observed_dice = [item.dice for item in summarize_classes(matrix.counts)]
    assert observed_dice == pytest.approx(expected_dice)


def test_absent_class_scores_are_undefined() -> None:
    matrix = PixelConfusionMatrix(NUM_CLASSES)
    matrix.update(torch.zeros(1, 2, 2, dtype=torch.long), torch.zeros(1, 2, 2, dtype=torch.long))
    assert all(math.isnan(item.dice) for item in summarize_classes(matrix.counts))


def test_dominant_tumor_label_ignores_background() -> None:
    assert dominant_tumor_label(build_mixed_prediction(), NUM_CLASSES) == 2
    assert dominant_tumor_label(torch.zeros(4, 4, dtype=torch.long), NUM_CLASSES) == 0


def test_majority_vote_relabels_tumor_pixels_per_slice() -> None:
    predictions = torch.stack([build_mixed_prediction(), torch.zeros(4, 4, dtype=torch.long)])
    voted = majority_vote(predictions, NUM_CLASSES)
    assert set(voted[0].unique().tolist()) == {0, 2}
    assert torch.equal(voted[0] != 0, predictions[0] != 0)
    assert torch.equal(voted[1], predictions[1])


def test_slice_record_scores() -> None:
    record = build_slice_record("1_1", build_mixed_prediction(), build_glioma_slice(), NUM_CLASSES)
    assert record.target_class == 2
    assert record.predicted_class == 2
    assert record.class_dice == pytest.approx(4 / 7)
    assert record.localization_dice == pytest.approx(6 / 8)
    assert record.majority_vote_dice == pytest.approx(record.localization_dice)
    assert record.has_tumor_prediction
    assert not record.has_wrong_type


def test_slice_record_with_wrong_type_scores_zero_after_vote() -> None:
    prediction = torch.zeros(4, 4, dtype=torch.long)
    prediction[:2, :2] = 3
    record = build_slice_record("1_2", prediction, build_glioma_slice(), NUM_CLASSES)
    assert record.has_wrong_type
    assert record.localization_dice == pytest.approx(1.0)
    assert record.class_dice == pytest.approx(0.0)
    assert record.majority_vote_dice == pytest.approx(0.0)


def test_slice_record_without_tumor_prediction() -> None:
    record = build_slice_record(
        "1_3", torch.zeros(4, 4, dtype=torch.long), build_glioma_slice(), NUM_CLASSES
    )
    assert record.predicted_class == 0
    assert not record.has_tumor_prediction
    assert not record.has_wrong_type
    assert record.class_dice == pytest.approx(0.0)


def test_build_slice_records_rejects_mismatched_names() -> None:
    with pytest.raises(ValueError):
        build_slice_records(["only_one"], torch.zeros(2, 4, 4), torch.zeros(2, 4, 4))


def test_slice_confusion_counts_majority_types() -> None:
    wrong_prediction = torch.zeros(4, 4, dtype=torch.long)
    wrong_prediction[:2, :2] = 3
    records = build_slice_records(
        ["a", "b", "c"],
        torch.stack(
            [build_mixed_prediction(), wrong_prediction, torch.zeros(4, 4, dtype=torch.long)]
        ),
        torch.stack([build_glioma_slice()] * 3),
        NUM_CLASSES,
    )
    slice_counts = build_slice_confusion(records, NUM_CLASSES)
    assert slice_counts.shape == (NUM_CLASSES - 1, NUM_CLASSES)
    assert slice_counts[1].tolist() == [1, 0, 1, 1]


def test_size_terciles_cover_every_record() -> None:
    records = [
        build_slice_record(str(index), build_mixed_prediction(), build_glioma_slice(), NUM_CLASSES)
        for index in range(7)
    ]
    terciles = split_into_size_terciles(records)
    assert sum(len(group) for group in terciles.values()) == 7
    assert list(terciles) == ["small", "medium", "large"]


def test_slice_group_summary() -> None:
    wrong_prediction = torch.zeros(4, 4, dtype=torch.long)
    wrong_prediction[:2, :2] = 3
    records = build_slice_records(
        ["a", "b"],
        torch.stack([build_glioma_slice(), wrong_prediction]),
        torch.stack([build_glioma_slice()] * 2),
        NUM_CLASSES,
    )
    summary = summarize_slice_group(records)
    assert summary.slices == 2
    assert summary.mean_slice_dice == pytest.approx(0.5)
    assert summary.low_dice_share == pytest.approx(0.5)
    assert summary.wrong_type_share == pytest.approx(0.5)
    assert summary.no_tumor_predicted_share == pytest.approx(0.0)
    assert summary.localization_dice == pytest.approx(1.0)
