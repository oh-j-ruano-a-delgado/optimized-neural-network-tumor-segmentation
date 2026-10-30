"""Train a U-Net baseline on the Figshare brain tumor dataset.

The data pipeline and the loss are selectable so every Phase 5 ablation runs through the same
loop: ``--augmentation`` picks the preset from ``src/dataset/transforms.py`` (none, basic,
advanced) and ``--loss`` picks the criterion from ``src/models/losses.py`` (dice_focal,
focal_tversky_focal, dice_ce). ``--region-weight`` weights the Dice or Focal Tversky term and
``--pixel-weight`` the Focal or cross-entropy term; the former names ``--dice-weight`` and
``--focal-weight`` remain accepted as aliases. The defaults (basic augmentation, Dice-Focal
0.5 / 0.5, focal gamma 2) reproduce the Phase 2-4 setup, and every argument is stored in the
checkpoint, so each run records which pipeline and loss produced it.

``--early-stopping-patience N`` stops training once the mean validation Dice has not improved for
N consecutive epochs; the checkpoint always holds the best epoch, so ``--epochs`` becomes an upper
bound. It is disabled by default, which keeps the fixed-length runs of earlier phases reproducible.

``--lr-schedule`` selects the learning-rate schedule from ``src/utils/training.py``: "constant"
(default, the Phase 2-5 behaviour) or "cosine", which anneals the rate from ``--learning-rate``
to zero over exactly ``--epochs`` epochs (``CosineAnnealingLR(optimizer, T_max=epochs)``, stepped
once per epoch). The rate used in each epoch is written to the ``learning_rate`` column of
``history.csv`` so the schedule can be verified after a run.

``--model`` selects the network from ``src/models/factory.py``: "unet" (default, the custom U-Net
sized by ``--base-filters`` and ``--depth``) or "smp_unet" (the segmentation_models_pytorch U-Net
with the pretrained encoder named by ``--encoder``, default resnet34, Phase 7). Options that do
not apply to the selected network are ignored. ``--encoder-weights`` is "imagenet" (default) or
"none"; "none" trains the same SMP network from random initialization, the control that separates
the effect of pretraining from that of the architecture. ``--encoder-lr-multiplier`` scales the
learning rate of the pretrained encoder relative to ``--learning-rate``, which keeps applying to
the decoder and segmentation head. The default of 1 trains the whole model at one rate, as in
Phases 2-6, and only "smp_unet" accepts other values. The checkpoint config records the
architecture, so the evaluation scripts rebuild either network from it.

Usage (from the project root):
    python -m scripts.train_baseline --epochs 40 [--device auto|cpu|cuda] [--amp] [--output-dir runs/baseline]
        [--augmentation none|basic|advanced_spatial|advanced_intensity|advanced]
        [--loss dice_focal|focal_tversky_focal|dice_ce]
        [--tversky-alpha 0.7] [--tversky-beta 0.3] [--tversky-gamma 1.3333]
        [--early-stopping-patience 20] [--lr-schedule constant|cosine]
        [--model unet|smp_unet] [--encoder resnet34] [--encoder-weights imagenet|none]
        [--encoder-lr-multiplier 0.1]
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Final, List, Sequence

import torch
from torch import nn

from src.dataset.constants import NUM_CLASSES, TUMOR_CLASS_NAMES
from src.dataset.dataloader import get_dataloaders
from src.dataset.transforms import AUGMENTATION_PRESETS, DEFAULT_AUGMENTATION_PRESET
from src.models.factory import (
    DEFAULT_ENCODER_NAME,
    DEFAULT_ENCODER_WEIGHTS,
    DEFAULT_MODEL_ARCHITECTURE,
    ENCODER_WEIGHTS,
    MODEL_ARCHITECTURES,
    ModelSettings,
    build_model,
    describe_model,
    model_config,
    resolve_encoder_weights,
)
from src.models.losses import (
    DEFAULT_DICE_WEIGHT,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_FOCAL_WEIGHT,
    DEFAULT_LOSS_NAME,
    DEFAULT_TVERSKY_ALPHA,
    DEFAULT_TVERSKY_BETA,
    DEFAULT_TVERSKY_GAMMA,
    LOSS_NAMES,
    LossSettings,
    build_criterion,
)
from src.models.unet import DEFAULT_IN_CHANNELS, count_parameters
from src.utils.metrics import SegmentationMetrics, SegmentationReport
from src.utils.training import (
    build_grad_scaler,
    evaluate_epoch,
    resolve_amp_mode,
    resolve_device,
    DEFAULT_ENCODER_LR_MULTIPLIER,
    DEFAULT_LEARNING_RATE_SCHEDULE,
    LEARNING_RATE_SCHEDULES,
    build_learning_rate_scheduler,
    build_optimizer,
    current_learning_rate,
    resolve_max_batches,
    set_global_seed,
    should_stop_early,
    train_epoch,
    validate_early_stopping_patience,
    validate_encoder_lr_multiplier,
)

DEFAULT_EPOCHS: Final[int] = 40
DEFAULT_BATCH_SIZE: Final[int] = 8
DEFAULT_LEARNING_RATE: Final[float] = 1e-3
DEFAULT_WEIGHT_DECAY: Final[float] = 1e-4
DEFAULT_IMAGE_SIZE: Final[int] = 256
DEFAULT_SEED: Final[int] = 42
DEFAULT_OUTPUT_DIR: Final[Path] = Path("runs/baseline")
ELAPSED_COLUMN: Final[str] = "elapsed_seconds"
LEARNING_RATE_COLUMN: Final[str] = "learning_rate"
MEAN_COLUMNS: Final[Dict[str, str]] = {"dice": "val_dice_mean", "iou": "val_iou_mean"}
PER_CLASS_COLUMNS: Final[Dict[str, str]] = {"dice": "val_dice", "iou": "val_iou"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--model", choices=MODEL_ARCHITECTURES, default=DEFAULT_MODEL_ARCHITECTURE)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER_NAME)
    parser.add_argument(
        "--encoder-weights", choices=ENCODER_WEIGHTS, default=DEFAULT_ENCODER_WEIGHTS
    )
    parser.add_argument(
        "--encoder-lr-multiplier", type=float, default=DEFAULT_ENCODER_LR_MULTIPLIER
    )
    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--augmentation", choices=AUGMENTATION_PRESETS, default=DEFAULT_AUGMENTATION_PRESET
    )
    parser.add_argument("--loss", choices=LOSS_NAMES, default=DEFAULT_LOSS_NAME)
    parser.add_argument(
        "--region-weight",
        "--dice-weight",
        dest="region_weight",
        type=float,
        default=DEFAULT_DICE_WEIGHT,
    )
    parser.add_argument(
        "--pixel-weight",
        "--focal-weight",
        dest="pixel_weight",
        type=float,
        default=DEFAULT_FOCAL_WEIGHT,
    )
    parser.add_argument("--focal-gamma", type=float, default=DEFAULT_FOCAL_GAMMA)
    parser.add_argument("--tversky-alpha", type=float, default=DEFAULT_TVERSKY_ALPHA)
    parser.add_argument("--tversky-beta", type=float, default=DEFAULT_TVERSKY_BETA)
    parser.add_argument("--tversky-gamma", type=float, default=DEFAULT_TVERSKY_GAMMA)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument(
        "--lr-schedule", choices=LEARNING_RATE_SCHEDULES, default=DEFAULT_LEARNING_RATE_SCHEDULE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def build_model_settings(arguments: argparse.Namespace) -> ModelSettings:
    return ModelSettings(
        architecture=arguments.model,
        in_channels=DEFAULT_IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_filters=arguments.base_filters,
        depth=arguments.depth,
        encoder_name=arguments.encoder,
    )


def build_loss_settings(arguments: argparse.Namespace) -> LossSettings:
    return LossSettings(
        name=arguments.loss,
        region_weight=arguments.region_weight,
        pixel_weight=arguments.pixel_weight,
        focal_gamma=arguments.focal_gamma,
        tversky_alpha=arguments.tversky_alpha,
        tversky_beta=arguments.tversky_beta,
        tversky_gamma=arguments.tversky_gamma,
    )


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
        LEARNING_RATE_COLUMN,
    ]


def build_history_row(
    epoch: int,
    train_loss: float,
    validation_loss: float,
    report: SegmentationReport,
    elapsed_seconds: float,
    learning_rate: float,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
        "val_loss": round(validation_loss, 6),
        MEAN_COLUMNS["dice"]: round(report.mean_dice, 6),
        MEAN_COLUMNS["iou"]: round(report.mean_iou, 6),
        ELAPSED_COLUMN: round(elapsed_seconds, 3),
        LEARNING_RATE_COLUMN: float(f"{learning_rate:.6g}"),
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
    path: Path,
    model: nn.Module,
    model_settings: ModelSettings,
    arguments: argparse.Namespace,
    epoch: int,
    validation_dice: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model_config(model_settings),
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
        f"lr {row[LEARNING_RATE_COLUMN]:.2e} | {row[ELAPSED_COLUMN]:.1f}s"
    )


def main() -> None:
    arguments = parse_arguments()
    validate_early_stopping_patience(arguments.early_stopping_patience)
    validate_encoder_lr_multiplier(arguments.encoder_lr_multiplier)
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
        augmentation=arguments.augmentation,
    )
    print(
        f"Device: {device} | AMP: {'on' if amp_enabled else 'off'} | "
        f"augmentation {arguments.augmentation} | loss {arguments.loss} | "
        f"lr schedule {arguments.lr_schedule} | "
        f"train {len(dataloaders.train.dataset)} samples | "
        f"validation {len(dataloaders.validation.dataset)} samples | "
        f"test {len(dataloaders.test.dataset)} samples"
    )

    model_settings = build_model_settings(arguments)
    encoder_weights = resolve_encoder_weights(arguments.model, arguments.encoder_weights)
    model = build_model(model_settings, encoder_weights).to(device)
    print(
        f"Model: {describe_model(model_settings)} | "
        f"pretrained encoder weights {encoder_weights or 'none'} | "
        f"encoder lr multiplier {arguments.encoder_lr_multiplier:g}"
    )
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = build_criterion(build_loss_settings(arguments), NUM_CLASSES).to(device)
    optimizer = build_optimizer(
        model, arguments.learning_rate, arguments.weight_decay, arguments.encoder_lr_multiplier
    )
    scheduler = build_learning_rate_scheduler(optimizer, arguments.lr_schedule, arguments.epochs)
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)

    class_names = [TUMOR_CLASS_NAMES[index] for index in metrics.class_indices]
    fieldnames = build_history_fieldnames(class_names)
    history: List[Dict[str, object]] = []
    best_dice = -1.0
    best_epoch = 0
    checkpoint_path = output_dir / "best_model.pth"

    for epoch in range(1, arguments.epochs + 1):
        start_time = time.perf_counter()
        epoch_learning_rate = current_learning_rate(optimizer)
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
        if scheduler is not None:
            scheduler.step()
        row = build_history_row(
            epoch,
            train_loss,
            validation_loss,
            report,
            time.perf_counter() - start_time,
            epoch_learning_rate,
        )
        history.append(row)
        write_history_csv(output_dir / "history.csv", fieldnames, history)
        print(describe_history_row(row, arguments.epochs))

        if report.mean_dice > best_dice:
            best_dice = report.mean_dice
            best_epoch = epoch
            save_checkpoint(checkpoint_path, model, model_settings, arguments, epoch, best_dice)

        if should_stop_early(epoch, best_epoch, arguments.early_stopping_patience):
            print(
                f"Early stopping at epoch {epoch}: no improvement in mean validation Dice for "
                f"{arguments.early_stopping_patience} epochs."
            )
            break

    print(
        f"Best validation Dice: {best_dice:.4f} (epoch {best_epoch}) | "
        f"checkpoint saved to '{checkpoint_path}'."
    )


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(f"Error: {error}")
