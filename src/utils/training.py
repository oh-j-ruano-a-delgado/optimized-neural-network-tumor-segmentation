"""Shared training-loop utilities: seeding, device and AMP setup, epochs and early stopping.

``train_epoch`` and ``evaluate_epoch`` run one pass over a DataLoader, optionally under CUDA
automatic mixed precision. The loss is always computed on logits cast to float32, because the
small smoothing epsilon of the Dice and Tversky losses would underflow in half precision.

Early stopping monitors the same quantity that selects the checkpoint (mean validation Dice over
the tumor classes): ``should_stop_early`` returns True once ``patience`` epochs have passed since
the best epoch, so the saved checkpoint is always the best one seen and training never continues
for more than ``patience`` epochs without a new best. A patience of None disables it.

Learning-rate schedules: "constant" keeps the optimizer's learning rate fixed (the Phase 2-5
setup) and returns no scheduler. "cosine" returns ``CosineAnnealingLR(optimizer,
T_max=total_epochs)``, stepped once per epoch after the training pass, so the rate decays from
its initial value to zero along half a cosine period that ends exactly at the last epoch. Late
epochs therefore take small steps, which damps the epoch-to-epoch swings in validation Dice that
a constant rate produces. If early stopping ends a run before ``total_epochs``, the cosine curve
is left unfinished, so cosine runs should normally disable early stopping.

Optimizer: ``build_optimizer`` creates the Adam optimizer of every run. With an encoder
learning-rate multiplier of 1 (the default) it holds a single parameter group, exactly as in
Phases 2-6. Any other multiplier requires a model with a pretrained encoder (``--model smp_unet``)
and creates two groups with the same weight decay: the randomly initialized decoder and
segmentation head at the base learning rate (group 0), and the pretrained encoder at the base rate
times the multiplier (group 1). A multiplier below 1 lets the new layers learn at full speed while
the ImageNet features change slowly, so the large, noisy gradients of the untrained decoder in the
first epochs cannot overwrite them. The cosine schedule anneals each group from its own initial
rate, so the encoder rate stays the multiplier times the decoder rate for the whole run, and
``current_learning_rate`` (group 0, the ``learning_rate`` column of ``history.csv``) keeps
reporting the base rate.
"""

import math
import random
from typing import Final, Literal, Optional, Tuple, get_args

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

from ..dataset.dataloader import SegmentationBatch
from ..models.factory import split_encoder_parameters
from .metrics import SegmentationMetrics, SegmentationReport

LearningRateSchedule = Literal["constant", "cosine"]
LEARNING_RATE_SCHEDULES: Final[Tuple[str, ...]] = get_args(LearningRateSchedule)
DEFAULT_LEARNING_RATE_SCHEDULE: Final[LearningRateSchedule] = "constant"
DEFAULT_ENCODER_LR_MULTIPLIER: Final[float] = 1.0


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except RuntimeError as error:
        raise ValueError(f"Unknown device '{requested}'.") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available on this machine.")
    return device


def resolve_amp_mode(amp_requested: bool, device: torch.device) -> bool:
    if amp_requested and device.type != "cuda":
        print("Mixed precision (--amp) requires CUDA; training in full precision.")
        return False
    return amp_requested


def build_grad_scaler(amp_enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=amp_enabled)


def validate_early_stopping_patience(patience: Optional[int]) -> None:
    if patience is not None and patience < 1:
        raise ValueError(f"Early stopping patience must be at least 1, got {patience}.")


def should_stop_early(current_epoch: int, best_epoch: int, patience: Optional[int]) -> bool:
    validate_early_stopping_patience(patience)
    return patience is not None and current_epoch - best_epoch >= patience


def build_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer, schedule: LearningRateSchedule, total_epochs: int
) -> Optional[LRScheduler]:
    if schedule not in LEARNING_RATE_SCHEDULES:
        raise ValueError(
            f"Unknown learning-rate schedule '{schedule}', expected one of "
            f"{LEARNING_RATE_SCHEDULES}."
        )
    if total_epochs < 1:
        raise ValueError(f"total_epochs must be at least 1, got {total_epochs}.")
    if schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_epochs)
    return None


def validate_encoder_lr_multiplier(multiplier: float) -> None:
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(
            f"The encoder learning-rate multiplier must be positive, got {multiplier}."
        )


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    encoder_lr_multiplier: float = DEFAULT_ENCODER_LR_MULTIPLIER,
) -> torch.optim.Adam:
    validate_encoder_lr_multiplier(encoder_lr_multiplier)
    if encoder_lr_multiplier == DEFAULT_ENCODER_LR_MULTIPLIER:
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    encoder_parameters, other_parameters = split_encoder_parameters(model)
    return torch.optim.Adam(
        [
            {"params": other_parameters, "lr": learning_rate},
            {"params": encoder_parameters, "lr": learning_rate * encoder_lr_multiplier},
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def resolve_max_batches(loader: DataLoader[SegmentationBatch], limit: Optional[int]) -> int:
    return len(loader) if limit is None else min(limit, len(loader))


def transfer_batch(
    images: Tensor, masks: Tensor, device: torch.device
) -> Tuple[Tensor, Tensor]:
    non_blocking = device.type == "cuda"
    return images.to(device, non_blocking=non_blocking), masks.to(
        device, non_blocking=non_blocking
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader[SegmentationBatch],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    device: torch.device,
    max_batches: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_index, (images, masks) in enumerate(loader):
        if batch_index >= max_batches:
            break
        images, masks = transfer_batch(images, masks, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits.float(), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]
    return total_loss / max(total_samples, 1)


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader[SegmentationBatch],
    criterion: nn.Module,
    metrics: SegmentationMetrics,
    amp_enabled: bool,
    device: torch.device,
    max_batches: int,
) -> Tuple[float, SegmentationReport]:
    model.eval()
    metrics.reset()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_index, (images, masks) in enumerate(loader):
            if batch_index >= max_batches:
                break
            images, masks = transfer_batch(images, masks, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits.float(), masks)
            total_loss += loss.item() * images.shape[0]
            total_samples += images.shape[0]
            metrics.update(logits, masks)
    return total_loss / max(total_samples, 1), metrics.compute()
