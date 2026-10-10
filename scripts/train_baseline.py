"""Train a U-Net baseline on the Figshare brain tumor dataset.

Usage (from the project root):
    python -m scripts.train_baseline --epochs 40 [--device auto|cpu|cuda] [--amp] [--output-dir runs/baseline]
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Final, List, Sequence

import torch

from src.dataset.constants import NUM_CLASSES, TUMOR_CLASS_NAMES
from src.dataset.dataloader import get_dataloaders
from src.models.losses import DiceFocalLoss
from src.models.unet import UNet, count_parameters
from src.utils.metrics import SegmentationMetrics, SegmentationReport
from src.utils.training import (
    build_grad_scaler,
    evaluate_epoch,
    resolve_amp_mode,
    resolve_device,
    resolve_max_batches,
    set_global_seed,
    train_epoch,
)

DEFAULT_EPOCHS: Final[int] = 40
DEFAULT_BATCH_SIZE: Final[int] = 8
DEFAULT_LEARNING_RATE: Final[float] = 1e-3
DEFAULT_WEIGHT_DECAY: Final[float] = 1e-4
DEFAULT_IMAGE_SIZE: Final[int] = 256
DEFAULT_SEED: Final[int] = 42
DEFAULT_OUTPUT_DIR: Final[Path] = Path("runs/baseline")
ELAPSED_COLUMN: Final[str] = "elapsed_seconds"
MEAN_COLUMNS: Final[Dict[str, str]] = {"dice": "val_dice_mean", "iou": "val_iou_mean"}
PER_CLASS_COLUMNS: Final[Dict[str, str]] = {"dice": "val_dice", "iou": "val_iou"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--focal-weight", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def build_history_fieldnames(class_names: Sequence[str]) -> List[str]:
    dice_fields = [f"{PER_CLASS_COLUMNS['dice']}_{name}" for name in class_names]
    iou_fields = [f"{PER_CLASS_COLUMNS['iou']}_{name}" for name in class_names]
    return [
        "epoch",
        "train_loss",
        "val_loss",
        MEAN_COLUMNS["dice"],
        *dice_fields,
        MEAN_COLUMNS["iou"],
        *iou_fields,
        ELAPSED_COLUMN,
    ]


def build_history_row(
    epoch: int,
    train_loss: float,
    validation_loss: float,
    report: SegmentationReport,
    elapsed_seconds: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
        "val_loss": round(validation_loss, 6),
        MEAN_COLUMNS["dice"]: round(report.mean_dice, 6),
        MEAN_COLUMNS["iou"]: round(report.mean_iou, 6),
        ELAPSED_COLUMN: round(elapsed_seconds, 3),
    }
    for metric_name, scores in (
        ("dice", report.dice_by_class),
        ("iou", report.iou_by_class),
    ):
        for class_index, score in zip(report.class_indices, scores.tolist()):
            class_name = TUMOR_CLASS_NAMES.get(class_index, "background")
            row[f"{PER_CLASS_COLUMNS[metric_name]}_{class_name}"] = round(score, 6)
    return row


def write_history_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path, model: UNet, arguments: argparse.Namespace, epoch: int, validation_dice: float
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "in_channels": model.in_channels,
                "num_classes": model.num_classes,
                "base_filters": model.base_filters,
                "depth": model.depth,
            },
            "epoch": epoch,
            "validation_dice": validation_dice,
            "arguments": vars(arguments),
        },
        path,
    )


def describe_history_row(row: Dict[str, object], total_epochs: int) -> str:
    return (
        f"Epoch {row['epoch']:>3}/{total_epochs} | "
        f"train loss {row['train_loss']:.4f} | val loss {row['val_loss']:.4f} | "
        f"val dice {row[MEAN_COLUMNS['dice']]:.4f} | val iou {row[MEAN_COLUMNS['iou']]:.4f} | "
        f"{row[ELAPSED_COLUMN]:.1f}s"
    )


def main() -> None:
    arguments = parse_arguments()
    set_global_seed(arguments.seed)
    device = resolve_device(arguments.device)
    amp_enabled = resolve_amp_mode(arguments.amp, device)
    scaler = build_grad_scaler(amp_enabled)
    output_dir: Path = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloaders = get_dataloaders(
        batch_size=arguments.batch_size,
        image_size=(arguments.image_size, arguments.image_size),
        validation_fraction=arguments.validation_fraction,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
    )
    print(
        f"Device: {device} | AMP: {'on' if amp_enabled else 'off'} | "
        f"train {len(dataloaders.train.dataset)} samples | "
        f"validation {len(dataloaders.validation.dataset)} samples | "
        f"test {len(dataloaders.test.dataset)} samples"
    )

    model = UNet(
        num_classes=NUM_CLASSES,
        base_filters=arguments.base_filters,
        depth=arguments.depth,
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = DiceFocalLoss(
        dice_weight=arguments.dice_weight,
        focal_weight=arguments.focal_weight,
        num_classes=NUM_CLASSES,
        gamma=arguments.focal_gamma,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay
    )
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)

    class_names = [TUMOR_CLASS_NAMES[index] for index in metrics.class_indices]
    fieldnames = build_history_fieldnames(class_names)
    history: List[Dict[str, object]] = []
    best_dice = -1.0
    checkpoint_path = output_dir / "best_model.pth"

    for epoch in range(1, arguments.epochs + 1):
        start_time = time.perf_counter()
        train_loss = train_epoch(
            model,
            dataloaders.train,
            criterion,
            optimizer,
            scaler,
            amp_enabled,
            device,
            resolve_max_batches(dataloaders.train, arguments.limit_batches),
        )
        validation_loss, report = evaluate_epoch(
            model,
            dataloaders.validation,
            criterion,
            metrics,
            amp_enabled,
            device,
            resolve_max_batches(dataloaders.validation, arguments.limit_batches),
        )
        row = build_history_row(
            epoch, train_loss, validation_loss, report, time.perf_counter() - start_time
        )
        history.append(row)
        write_history_csv(output_dir / "history.csv", fieldnames, history)
        print(describe_history_row(row, arguments.epochs))

        if report.mean_dice > best_dice:
            best_dice = report.mean_dice
            save_checkpoint(checkpoint_path, model, arguments, epoch, best_dice)

    print(f"Best validation Dice: {best_dice:.4f} | checkpoint saved to '{checkpoint_path}'.")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(f"Error: {error}")
