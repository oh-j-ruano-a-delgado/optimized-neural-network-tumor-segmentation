"""Tests for the shared training-loop utilities in ``src.utils.training``.

The epoch tests train a real U-Net on a tiny random dataset (4 images of 32 x 32) to check that a
training pass lowers the loss and an evaluation pass yields finite metrics. The early-stopping
tests pin down the patience semantics: with patience N, training stops at the first epoch that is
N epochs after the best one, never earlier, and a missing patience disables early stopping.
"""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.losses import DiceFocalLoss
from src.models.unet import UNet
from src.utils.metrics import SegmentationMetrics
from src.utils.training import (
    build_grad_scaler,
    evaluate_epoch,
    resolve_amp_mode,
    resolve_device,
    set_global_seed,
    should_stop_early,
    train_epoch,
    validate_early_stopping_patience,
)


def build_tiny_loader(seed: int = 0) -> DataLoader:
    torch.manual_seed(seed)
    images = torch.rand(4, 1, 32, 32)
    masks = (torch.rand(4, 32, 32) > 0.5).long() * torch.randint(1, 4, (4, 1, 1))
    return DataLoader(TensorDataset(images, masks), batch_size=2)


def test_resolve_device_returns_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_matches_availability() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected


def test_resolve_device_rejects_unavailable_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("machine has CUDA")
    with pytest.raises(ValueError, match="CUDA"):
        resolve_device("cuda")


def test_resolve_device_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("banana")


def test_amp_is_disabled_on_cpu() -> None:
    assert resolve_amp_mode(True, torch.device("cpu")) is False
    assert resolve_amp_mode(False, torch.device("cpu")) is False


def test_amp_is_enabled_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("machine has CUDA")
    assert resolve_amp_mode(True, torch.device("cuda")) is True
    assert resolve_amp_mode(False, torch.device("cuda")) is False


def test_grad_scaler_is_disabled_when_amp_is_off() -> None:
    assert build_grad_scaler(False).is_enabled() is False


def test_evaluate_epoch_returns_finite_metrics() -> None:
    set_global_seed(0)
    loader = build_tiny_loader()
    model = UNet(base_filters=4, depth=2)
    criterion = DiceFocalLoss()
    loss, report = evaluate_epoch(
        model,
        loader,
        criterion,
        SegmentationMetrics(),
        amp_enabled=False,
        device=torch.device("cpu"),
        max_batches=100,
    )
    assert math.isfinite(loss)
    assert 0.0 <= report.mean_dice <= 1.0
    assert 0.0 <= report.mean_iou <= 1.0


def test_train_epoch_reduces_loss_when_overfitting() -> None:
    set_global_seed(0)
    loader = build_tiny_loader()
    model = UNet(base_filters=4, depth=2)
    criterion = DiceFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scaler = build_grad_scaler(False)
    losses = [
        train_epoch(
            model,
            loader,
            criterion,
            optimizer,
            scaler,
            amp_enabled=False,
            device=torch.device("cpu"),
            max_batches=100,
        )
        for _ in range(30)
    ]
    assert losses[-1] < losses[0]


def test_early_stopping_waits_for_the_full_patience() -> None:
    assert not should_stop_early(current_epoch=116, best_epoch=97, patience=20)
    assert should_stop_early(current_epoch=117, best_epoch=97, patience=20)


def test_early_stopping_never_triggers_on_the_best_epoch() -> None:
    assert not should_stop_early(current_epoch=50, best_epoch=50, patience=1)


def test_early_stopping_is_disabled_without_patience() -> None:
    assert not should_stop_early(current_epoch=500, best_epoch=1, patience=None)


@pytest.mark.parametrize("patience", [0, -3])
def test_early_stopping_rejects_non_positive_patience(patience: int) -> None:
    with pytest.raises(ValueError, match="patience"):
        validate_early_stopping_patience(patience)
    with pytest.raises(ValueError, match="patience"):
        should_stop_early(current_epoch=10, best_epoch=1, patience=patience)
