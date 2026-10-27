"""Shared training-loop utilities: seeding, device and AMP setup, epochs and early stopping.

``train_epoch`` and ``evaluate_epoch`` run one pass over a DataLoader, optionally under CUDA
automatic mixed precision. The loss is always computed on logits cast to float32, because the
small smoothing epsilon of the Dice and Tversky losses would underflow in half precision.

Early stopping monitors the same quantity that selects the checkpoint (mean validation Dice over
the tumor classes): ``should_stop_early`` returns True once ``patience`` epochs have passed since
the best epoch, so the saved checkpoint is always the best one seen and training never continues
for more than ``patience`` epochs without a new best. A patience of None disables it.
"""

import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..dataset.dataloader import SegmentationBatch
from .metrics import SegmentationMetrics, SegmentationReport


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
