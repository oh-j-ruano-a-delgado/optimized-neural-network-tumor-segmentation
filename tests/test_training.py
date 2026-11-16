"""Tests for the shared training-loop utilities in ``src.utils.training``.

The epoch tests train a real U-Net on a tiny random dataset (4 images of 32 x 32) to check that a
training pass lowers the loss and an evaluation pass yields finite metrics. The early-stopping
tests pin down the patience semantics: with patience N, training stops at the first epoch that is
N epochs after the best one, never earlier, and a missing patience disables early stopping.
The learning-rate schedule tests step a real Adam optimizer once per epoch and check the cosine
curve at its start, midpoint and end (base rate, half the base rate, zero) against the closed
form, and that the constant schedule returns no scheduler at all.
The optimizer tests check that the default multiplier keeps one parameter group, that any other
multiplier gives the pretrained encoder of the SMP U-Net (built offline, without ImageNet weights)
its own group at the scaled rate, that the cosine schedule keeps that ratio while annealing, and
that the custom U-Net or a non-positive multiplier is rejected.
"""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.factory import ModelSettings, build_model
from src.models.losses import DiceFocalLoss
from src.models.unet import UNet
from src.utils.metrics import SegmentationMetrics
from src.utils.training import (
    LEARNING_RATE_SCHEDULES,
    build_grad_scaler,
    build_learning_rate_scheduler,
    build_optimizer,
    current_learning_rate,
    evaluate_epoch,
    resolve_amp_mode,
    resolve_device,
    set_global_seed,
    should_stop_early,
    train_epoch,
    validate_early_stopping_patience,
    validate_encoder_lr_multiplier,
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


def build_single_parameter_optimizer(learning_rate: float = 1e-3) -> torch.optim.Optimizer:
    return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=learning_rate)


def test_constant_schedule_returns_no_scheduler() -> None:
    optimizer = build_single_parameter_optimizer()
    assert build_learning_rate_scheduler(optimizer, "constant", total_epochs=10) is None
    assert current_learning_rate(optimizer) == pytest.approx(1e-3)


def test_cosine_schedule_follows_half_a_cosine_period() -> None:
    base_rate, total_epochs = 1e-3, 150
    optimizer = build_single_parameter_optimizer(base_rate)
    scheduler = build_learning_rate_scheduler(optimizer, "cosine", total_epochs)
    rates = []
    for _ in range(total_epochs):
        rates.append(current_learning_rate(optimizer))
        optimizer.step()
        scheduler.step()
    expected = [
        0.5 * base_rate * (1.0 + math.cos(math.pi * epoch / total_epochs))
        for epoch in range(total_epochs)
    ]
    assert rates == pytest.approx(expected, rel=1e-6, abs=1e-12)
    assert rates[0] == pytest.approx(base_rate)
    assert rates[total_epochs // 2] == pytest.approx(base_rate / 2)
    assert current_learning_rate(optimizer) == pytest.approx(0.0, abs=1e-12)


def test_cosine_schedule_decreases_monotonically() -> None:
    optimizer = build_single_parameter_optimizer()
    scheduler = build_learning_rate_scheduler(optimizer, "cosine", total_epochs=20)
    rates = []
    for _ in range(20):
        rates.append(current_learning_rate(optimizer))
        optimizer.step()
        scheduler.step()
    assert all(earlier > later for earlier, later in zip(rates, rates[1:]))


def test_schedule_names_are_constant_and_cosine() -> None:
    assert LEARNING_RATE_SCHEDULES == ("constant", "cosine")


def test_unknown_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown learning-rate schedule"):
        build_learning_rate_scheduler(build_single_parameter_optimizer(), "step", total_epochs=10)


def test_schedule_rejects_non_positive_epochs() -> None:
    with pytest.raises(ValueError, match="total_epochs"):
        build_learning_rate_scheduler(build_single_parameter_optimizer(), "cosine", total_epochs=0)


def build_smp_unet() -> torch.nn.Module:
    return build_model(ModelSettings(architecture="smp_unet", encoder_name="resnet34"))


def test_default_optimizer_has_one_parameter_group() -> None:
    model = build_smp_unet()
    optimizer = build_optimizer(model, learning_rate=1e-3, weight_decay=1e-5)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-5)
    assert len(optimizer.param_groups[0]["params"]) == len(list(model.parameters()))


def test_encoder_multiplier_gives_the_encoder_a_scaled_learning_rate() -> None:
    model = build_smp_unet()
    optimizer = build_optimizer(
        model, learning_rate=3e-4, weight_decay=1e-5, encoder_lr_multiplier=0.1
    )
    decoder_group, encoder_group = optimizer.param_groups
    assert decoder_group["lr"] == pytest.approx(3e-4)
    assert encoder_group["lr"] == pytest.approx(3e-5)
    assert decoder_group["weight_decay"] == encoder_group["weight_decay"] == pytest.approx(1e-5)
    assert {id(parameter) for parameter in encoder_group["params"]} == {
        id(parameter) for parameter in model.encoder.parameters()
    }
    assert current_learning_rate(optimizer) == pytest.approx(3e-4)


def test_cosine_schedule_keeps_the_encoder_to_decoder_ratio() -> None:
    optimizer = build_optimizer(
        build_smp_unet(), learning_rate=3e-4, weight_decay=0.0, encoder_lr_multiplier=0.1
    )
    scheduler = build_learning_rate_scheduler(optimizer, "cosine", total_epochs=10)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    decoder_group, encoder_group = optimizer.param_groups
    assert 0.0 < decoder_group["lr"] < 3e-4
    assert encoder_group["lr"] == pytest.approx(0.1 * decoder_group["lr"])


def test_encoder_multiplier_is_rejected_for_the_custom_unet() -> None:
    with pytest.raises(ValueError, match="no pretrained encoder"):
        build_optimizer(
            UNet(base_filters=4, depth=2),
            learning_rate=1e-3,
            weight_decay=0.0,
            encoder_lr_multiplier=0.1,
        )


@pytest.mark.parametrize("multiplier", [0.0, -0.1, math.nan, math.inf])
def test_non_positive_or_non_finite_multiplier_is_rejected(multiplier: float) -> None:
    with pytest.raises(ValueError, match="multiplier"):
        validate_encoder_lr_multiplier(multiplier)
