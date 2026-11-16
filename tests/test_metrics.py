import math

import pytest
import torch

from src.utils.metrics import SegmentationMetrics


def test_hand_computed_dice_and_iou() -> None:
    target = torch.tensor([[[1, 1], [0, 0]]])
    prediction = torch.tensor([[[1, 0], [0, 0]]])
    metrics = SegmentationMetrics(num_classes=4)
    metrics.update(prediction, target)
    report = metrics.compute()
    class_one = report.class_indices.index(1)
    assert report.dice_by_class[class_one].item() == pytest.approx(2 / 3)
    assert report.iou_by_class[class_one].item() == pytest.approx(1 / 2)


def test_perfect_segmentation_scores_one() -> None:
    target = torch.tensor([[[1, 2], [1, 2]]])
    metrics = SegmentationMetrics(num_classes=4)
    metrics.update(target, target)
    report = metrics.compute()
    assert report.mean_dice == 1.0
    assert report.mean_iou == 1.0


def test_background_is_excluded_by_default() -> None:
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    metrics = SegmentationMetrics(num_classes=4)
    assert metrics.class_indices == [1, 2, 3]
    with_background = SegmentationMetrics(num_classes=4, include_background=True)
    assert with_background.class_indices == [0, 1, 2, 3]


def test_absent_classes_are_nan_and_ignored_by_the_mean() -> None:
    target = torch.tensor([[[1, 1], [0, 0]]])
    metrics = SegmentationMetrics(num_classes=4)
    metrics.update(target, target)
    report = metrics.compute()
    assert report.class_indices == (1, 2, 3)
    assert not math.isnan(report.dice_by_class[0].item())
    assert math.isnan(report.dice_by_class[1].item())
    assert math.isnan(report.dice_by_class[2].item())
    assert report.mean_dice == 1.0


def test_accumulator_aggregates_globally_across_batches() -> None:
    metrics = SegmentationMetrics(num_classes=4)
    perfect_target = torch.tensor([[[1, 1], [1, 1]]])
    missed_target = torch.tensor([[[1, 1], [1, 1]]])
    missed_prediction = torch.zeros_like(missed_target)
    metrics.update(perfect_target, perfect_target)
    metrics.update(missed_prediction, missed_target)
    report = metrics.compute()
    # Global accumulation gives dice 4/6 instead of the batch average (1+0)/2.
    assert report.dice_by_class[0].item() == pytest.approx(2 / 3)


def test_update_accepts_logits_and_hard_predictions() -> None:
    target = torch.tensor([[[1, 1], [0, 0]]])
    logits = torch.zeros(1, 4, 2, 2)
    logits[:, 1] = torch.where(target == 1, 10.0, -10.0)
    logits[:, 0] = torch.where(target == 1, -10.0, 10.0)
    from_logits = SegmentationMetrics(num_classes=4)
    from_logits.update(logits, target)
    from_predictions = SegmentationMetrics(num_classes=4)
    from_predictions.update(logits.argmax(dim=1), target)
    assert torch.allclose(
        from_logits.compute().dice_by_class,
        from_predictions.compute().dice_by_class,
        equal_nan=True,
    )


def test_reset_clears_accumulated_statistics() -> None:
    target = torch.tensor([[[1, 1], [0, 0]]])
    metrics = SegmentationMetrics(num_classes=4)
    metrics.update(target, target)
    metrics.reset()
    report = metrics.compute()
    assert all(math.isnan(score.item()) for score in report.dice_by_class)
    assert math.isnan(report.mean_dice)
