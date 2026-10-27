"""Diagnose where the GA-optimized U-Net loses Dice on the validation split (Phase 5, Step 0).

The checkpoint is rebuilt from its stored architecture and evaluated with the deterministic
evaluation transform at the image size it was trained with. The validation split is rebuilt with
the ``validation_fraction`` and ``seed`` stored in the checkpoint's CLI arguments, so it contains
exactly the patients the model was selected on; the recomputed mean Dice is printed next to the
stored validation Dice as a consistency check (small differences can come from mixed precision).
The validation split is the default on purpose: it informs design choices for the next training
runs, and the Testing split must stay untouched until the final comparison. ``--split test`` is
available for the final report only.

Reading the report:
    FN/FP ratio > 1 for a class means it loses more Dice by missing tumor pixels than by false
    alarms, which supports the Focal Tversky weights alpha = 0.7 (false negatives) and beta = 0.3
    (false positives); a ratio < 1 means the weights should be swapped.
    "Missed->bg" and "Confused" split each class's false negatives into pixels predicted as
    background (outline and recall errors, targeted by the new loss and augmentations) and pixels
    predicted as another tumor type (type confusion).
    "Dice (vote)" is the Dice after relabelling every predicted tumor pixel of a slice with the
    slice's majority tumor type. Figshare slices never mix tumor types, so a large gain over the
    raw Dice means a cheap post-processing step would recover it without retraining.
    The localization Dice merges all tumor types into one label. When it is close to the class
    Dice, the errors are outline errors; when it is much higher, they are type confusions.
    The size terciles show whether small tumors (lowest third of tumor area per class) fail more
    often, which matters for how strongly the spatial augmentations may shrink or warp the image.

Outputs (in ``--output-dir``):
    diagnostics.json        every metric above plus raw pixel and slice confusion matrices
    slice_diagnostics.csv   one row per slice with its pixel counts and Dice scores

Usage (from the project root):
    python -m scripts.evaluate_diagnostics [--checkpoint runs/optimized/best_model.pth]
        [--split validation|test] [--device auto|cpu|cuda] [--amp] [--batch-size 8]
        [--num-workers 0] [--output-dir runs/diagnostics]
"""

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.dataset.constants import (
    BACKGROUND_LABEL,
    NUM_CLASSES,
    RAW_DATA_DIR,
    TESTING_SPLIT,
    TRAINING_SPLIT,
    TUMOR_CLASS_NAMES,
)
from src.dataset.dataloader import (
    BrainTumorSegmentationDataset,
    SegmentationBatch,
    SegmentationSample,
    build_dataloader,
    discover_samples,
    split_samples_by_patient,
)
from src.dataset.transforms import build_evaluation_transform
from src.utils.checkpoint import (
    TrainingCheckpoint,
    build_model_from_checkpoint,
    load_training_checkpoint,
)
from src.utils.diagnostics import (
    ClassDiagnostics,
    PixelConfusionMatrix,
    SIZE_TERCILE_NAMES,
    SliceGroupSummary,
    SliceRecord,
    build_slice_confusion,
    build_slice_records,
    majority_vote,
    mean_of_defined,
    summarize_classes,
    summarize_slices_by_class,
)
from src.utils.training import resolve_amp_mode, resolve_device, transfer_batch

DEFAULT_CHECKPOINT: Final[Path] = Path("runs/optimized/best_model.pth")
DEFAULT_OUTPUT_DIR: Final[Path] = Path("runs/diagnostics")
DEFAULT_BATCH_SIZE: Final[int] = 8
DEFAULT_VALIDATION_FRACTION: Final[float] = 0.2
DEFAULT_SEED: Final[int] = 42
VALIDATION_SPLIT_NAME: Final[str] = "validation"
TEST_SPLIT_NAME: Final[str] = "test"
NO_TUMOR_NAME: Final[str] = "none"
BACKGROUND_NAME: Final[str] = "background"
REPORT_JSON_FILENAME: Final[str] = "diagnostics.json"
SLICE_CSV_FILENAME: Final[str] = "slice_diagnostics.csv"
SLICE_CSV_FIELDNAMES: Final[Tuple[str, ...]] = (
    "sample_name",
    "patient_id",
    "target_class",
    "predicted_class",
    "target_tumor_pixels",
    "predicted_tumor_pixels",
    "tumor_overlap_pixels",
    "class_dice",
    "localization_dice",
    "majority_vote_dice",
)
TUMOR_CLASS_INDICES: Final[Tuple[int, ...]] = tuple(sorted(TUMOR_CLASS_NAMES))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--split", choices=(VALIDATION_SPLIT_NAME, TEST_SPLIT_NAME), default=VALIDATION_SPLIT_NAME
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def class_name(class_index: int) -> str:
    if class_index == BACKGROUND_LABEL:
        return BACKGROUND_NAME
    return TUMOR_CLASS_NAMES[class_index]


def predicted_class_name(class_index: int) -> str:
    return NO_TUMOR_NAME if class_index == BACKGROUND_LABEL else TUMOR_CLASS_NAMES[class_index]


def finite_or_none(value: float, digits: int = 6) -> Optional[float]:
    return round(value, digits) if math.isfinite(value) else None


def resolve_split_settings(
    checkpoint: TrainingCheckpoint, arguments: argparse.Namespace
) -> Tuple[float, int]:
    validation_fraction = (
        arguments.validation_fraction
        if arguments.validation_fraction is not None
        else float(checkpoint.arguments.get("validation_fraction", DEFAULT_VALIDATION_FRACTION))
    )
    seed = (
        arguments.seed
        if arguments.seed is not None
        else int(checkpoint.arguments.get("seed", DEFAULT_SEED))
    )
    return validation_fraction, seed


def select_samples(
    split: str, data_dir: Path, validation_fraction: float, seed: int
) -> List[SegmentationSample]:
    if split == TEST_SPLIT_NAME:
        return discover_samples(data_dir / TESTING_SPLIT)
    _, validation_samples = split_samples_by_patient(
        discover_samples(data_dir / TRAINING_SPLIT), validation_fraction, seed
    )
    return validation_samples


def build_evaluation_loader(
    samples: Sequence[SegmentationSample],
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader[SegmentationBatch]:
    dataset = BrainTumorSegmentationDataset(
        samples, build_evaluation_transform((image_size, image_size))
    )
    return build_dataloader(
        dataset, batch_size, is_training=False, num_workers=num_workers, seed=seed
    )


def collect_diagnostics(
    model: nn.Module,
    loader: DataLoader[SegmentationBatch],
    samples: Sequence[SegmentationSample],
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[PixelConfusionMatrix, PixelConfusionMatrix, List[SliceRecord]]:
    raw_confusion = PixelConfusionMatrix(NUM_CLASSES)
    voted_confusion = PixelConfusionMatrix(NUM_CLASSES)
    records: List[SliceRecord] = []
    sample_names = [sample.image_path.stem for sample in samples]
    model.eval()
    with torch.inference_mode():
        for images, masks in loader:
            images, masks = transfer_batch(images, masks, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predictions = model(images).argmax(dim=1)
            predictions_cpu = predictions.cpu()
            masks_cpu = masks.cpu()
            raw_confusion.update(predictions_cpu, masks_cpu)
            voted_confusion.update(majority_vote(predictions_cpu, NUM_CLASSES), masks_cpu)
            batch_names = sample_names[len(records) : len(records) + masks_cpu.shape[0]]
            records.extend(
                build_slice_records(batch_names, predictions_cpu, masks_cpu, NUM_CLASSES)
            )
    if len(records) != len(samples):
        raise ValueError(f"Evaluated {len(records)} slices but the split has {len(samples)}.")
    return raw_confusion, voted_confusion, records


def serialize_class(
    diagnostics: ClassDiagnostics, voted_diagnostics: ClassDiagnostics
) -> Dict[str, Any]:
    return {
        "precision": finite_or_none(diagnostics.precision),
        "recall": finite_or_none(diagnostics.recall),
        "dice": finite_or_none(diagnostics.dice),
        "iou": finite_or_none(diagnostics.iou),
        "target_pixels": diagnostics.target_pixels,
        "predicted_pixels": diagnostics.predicted_pixels,
        "false_negative_pixels": diagnostics.false_negative_pixels,
        "false_positive_pixels": diagnostics.false_positive_pixels,
        "false_negative_to_false_positive_ratio": finite_or_none(
            diagnostics.false_negative_to_false_positive_ratio
        ),
        "missed_as_background_share": finite_or_none(diagnostics.missed_as_background_share),
        "confused_with_other_tumor_share": finite_or_none(
            diagnostics.confused_with_other_tumor_share
        ),
        "majority_vote_dice": finite_or_none(voted_diagnostics.dice),
    }


def serialize_slice_group(summary: SliceGroupSummary) -> Dict[str, Any]:
    return {
        "slices": summary.slices,
        "mean_slice_dice": finite_or_none(summary.mean_slice_dice),
        "median_slice_dice": finite_or_none(summary.median_slice_dice),
        "low_dice_share": finite_or_none(summary.low_dice_share),
        "no_tumor_predicted_share": finite_or_none(summary.no_tumor_predicted_share),
        "wrong_type_share": finite_or_none(summary.wrong_type_share),
        "localization_dice": finite_or_none(summary.localization_dice),
        "mean_slice_dice_by_size": {
            name: finite_or_none(value) for name, value in summary.mean_slice_dice_by_size.items()
        },
        "median_tumor_pixels_by_size": {
            name: finite_or_none(value, 1)
            for name, value in summary.median_tumor_pixels_by_size.items()
        },
    }


def build_report(
    checkpoint: TrainingCheckpoint,
    split: str,
    validation_fraction: float,
    seed: int,
    samples: Sequence[SegmentationSample],
    raw_confusion: PixelConfusionMatrix,
    voted_confusion: PixelConfusionMatrix,
    records: Sequence[SliceRecord],
    evaluation_seconds: float,
) -> Dict[str, Any]:
    raw_classes = summarize_classes(raw_confusion.counts)
    voted_classes = summarize_classes(voted_confusion.counts)
    slice_summaries = summarize_slices_by_class(records, TUMOR_CLASS_INDICES)
    slices_per_class = Counter(sample.class_label for sample in samples)
    return {
        "checkpoint": checkpoint.path.as_posix(),
        "checkpoint_epoch": checkpoint.epoch,
        "stored_validation_dice": finite_or_none(checkpoint.validation_dice),
        "hyperparameters": checkpoint.hyperparameters,
        "image_size": checkpoint.image_size,
        "split": split,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "slices": len(samples),
        "slices_per_class": {
            TUMOR_CLASS_NAMES[index]: slices_per_class[index] for index in TUMOR_CLASS_INDICES
        },
        "evaluation_seconds": round(evaluation_seconds, 2),
        "mean_dice": finite_or_none(mean_of_defined([item.dice for item in raw_classes])),
        "mean_dice_majority_vote": finite_or_none(
            mean_of_defined([item.dice for item in voted_classes])
        ),
        "classes": {
            class_name(raw.class_index): serialize_class(raw, voted)
            for raw, voted in zip(raw_classes, voted_classes)
        },
        "pixel_confusion_matrix": {
            "rows_ground_truth": [class_name(index) for index in range(NUM_CLASSES)],
            "columns_prediction": [class_name(index) for index in range(NUM_CLASSES)],
            "counts": raw_confusion.counts.tolist(),
        },
        "slice_confusion_matrix": {
            "rows_ground_truth": [TUMOR_CLASS_NAMES[index] for index in TUMOR_CLASS_INDICES],
            "columns_majority_prediction": [
                predicted_class_name(index) for index in range(NUM_CLASSES)
            ],
            "counts": build_slice_confusion(records, NUM_CLASSES).tolist(),
        },
        "slices_by_class": {
            TUMOR_CLASS_NAMES[index]: serialize_slice_group(summary)
            for index, summary in slice_summaries.items()
        },
    }


def format_score(value: Optional[float], width: int = 10, percent: bool = False) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.1%}" if percent else f"{value:>{width}.4f}"


def format_class_table(report: Mapping[str, Any]) -> List[str]:
    header = (
        f"{'Class':<12}{'Precision':>10}{'Recall':>10}{'Dice':>10}{'IoU':>10}{'FN/FP':>10}"
        f"{'Missed->bg':>12}{'Confused':>10}{'Dice(vote)':>12}"
    )
    lines = [header, "-" * len(header)]
    for name, scores in report["classes"].items():
        lines.append(
            f"{name:<12}{format_score(scores['precision'])}{format_score(scores['recall'])}"
            f"{format_score(scores['dice'])}{format_score(scores['iou'])}"
            f"{format_score(scores['false_negative_to_false_positive_ratio'])}"
            f"{format_score(scores['missed_as_background_share'], 12, percent=True)}"
            f"{format_score(scores['confused_with_other_tumor_share'], percent=True)}"
            f"{format_score(scores['majority_vote_dice'], 12)}"
        )
    lines.append(
        f"{'mean':<12}{'':>20}{format_score(report['mean_dice'])}{'':>42}"
        f"{format_score(report['mean_dice_majority_vote'], 12)}"
    )
    return lines


def row_percentages(counts: Tensor) -> Tensor:
    row_totals = counts.sum(dim=1, keepdim=True).clamp_min(1)
    return counts.double() / row_totals


def format_matrix(
    title: str, row_names: Sequence[str], column_names: Sequence[str], counts: Tensor
) -> List[str]:
    percentages = row_percentages(counts)
    header = f"{'':<12}" + "".join(f"{name:>12}" for name in column_names)
    lines = [title, header]
    for row_name, row in zip(row_names, percentages.tolist()):
        lines.append(f"{row_name:<12}" + "".join(f"{value:>12.1%}" for value in row))
    return lines


def format_slice_table(report: Mapping[str, Any]) -> List[str]:
    size_headers = "".join(f"{name.capitalize():>9}" for name in SIZE_TERCILE_NAMES)
    header = (
        f"{'Class':<12}{'Slices':>7}{'Mean':>8}{'Median':>8}{'<0.5':>8}{'NoTumor':>9}"
        f"{'WrongType':>11}{'LocDice':>9}{size_headers}"
    )
    lines = [header, "-" * len(header)]
    for name, summary in report["slices_by_class"].items():
        size_scores = "".join(
            format_score(summary["mean_slice_dice_by_size"][size], 9) for size in SIZE_TERCILE_NAMES
        )
        lines.append(
            f"{name:<12}{summary['slices']:>7}{format_score(summary['mean_slice_dice'], 8)}"
            f"{format_score(summary['median_slice_dice'], 8)}"
            f"{format_score(summary['low_dice_share'], 8, percent=True)}"
            f"{format_score(summary['no_tumor_predicted_share'], 9, percent=True)}"
            f"{format_score(summary['wrong_type_share'], 11, percent=True)}"
            f"{format_score(summary['localization_dice'], 9)}{size_scores}"
        )
    return lines


def format_report(report: Mapping[str, Any]) -> List[str]:
    pixel_matrix = report["pixel_confusion_matrix"]
    slice_matrix = report["slice_confusion_matrix"]
    return [
        "Per-class pixel metrics (global accumulation over the split)",
        *format_class_table(report),
        "",
        *format_matrix(
            "Pixel confusion matrix (rows = ground truth, columns = prediction, % of row)",
            pixel_matrix["rows_ground_truth"],
            pixel_matrix["columns_prediction"],
            torch.tensor(pixel_matrix["counts"]),
        ),
        "",
        *format_matrix(
            "Slice tumor type (rows = ground truth, columns = majority predicted type, % of row)",
            slice_matrix["rows_ground_truth"],
            slice_matrix["columns_majority_prediction"],
            torch.tensor(slice_matrix["counts"]),
        ),
        "",
        "Per-slice Dice of the ground-truth class (size columns = mean Dice per tumor-area third)",
        *format_slice_table(report),
    ]


def write_slice_csv(
    path: Path, records: Sequence[SliceRecord], samples: Sequence[SegmentationSample]
) -> None:
    patient_by_name = {sample.image_path.stem: sample.patient_id for sample in samples}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SLICE_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            {
                "sample_name": record.sample_name,
                "patient_id": patient_by_name.get(record.sample_name, ""),
                "target_class": predicted_class_name(record.target_class),
                "predicted_class": predicted_class_name(record.predicted_class),
                "target_tumor_pixels": record.target_tumor_pixels,
                "predicted_tumor_pixels": record.predicted_tumor_pixels,
                "tumor_overlap_pixels": record.tumor_overlap_pixels,
                "class_dice": finite_or_none(record.class_dice),
                "localization_dice": finite_or_none(record.localization_dice),
                "majority_vote_dice": finite_or_none(record.majority_vote_dice),
            }
            for record in records
        )


def describe_run(report: Mapping[str, Any], device: torch.device, amp_enabled: bool) -> List[str]:
    composition = ", ".join(f"{name} {count}" for name, count in report["slices_per_class"].items())
    stored_dice = report["stored_validation_dice"]
    return [
        f"Device: {device} | AMP: {'on' if amp_enabled else 'off'} | split {report['split']}: "
        f"{report['slices']} slices ({composition}) | validation fraction "
        f"{report['validation_fraction']} | seed {report['seed']}",
        f"Checkpoint: {report['checkpoint']} | epoch {report['checkpoint_epoch']} | "
        f"stored validation Dice {format_score(stored_dice, 0)} | "
        f"recomputed mean Dice {format_score(report['mean_dice'], 0)} | "
        f"evaluated in {report['evaluation_seconds']:.1f}s",
    ]


def main() -> None:
    arguments = parse_arguments()
    device = resolve_device(arguments.device)
    amp_enabled = resolve_amp_mode(arguments.amp, device)
    checkpoint = load_training_checkpoint(arguments.checkpoint)
    validation_fraction, seed = resolve_split_settings(checkpoint, arguments)
    samples = select_samples(arguments.split, arguments.data_dir, validation_fraction, seed)
    model = build_model_from_checkpoint(checkpoint, device)
    loader = build_evaluation_loader(
        samples, checkpoint.image_size, arguments.batch_size, arguments.num_workers, seed
    )

    start_time = time.perf_counter()
    raw_confusion, voted_confusion, records = collect_diagnostics(
        model, loader, samples, device, amp_enabled
    )
    report = build_report(
        checkpoint,
        arguments.split,
        validation_fraction,
        seed,
        samples,
        raw_confusion,
        voted_confusion,
        records,
        time.perf_counter() - start_time,
    )

    print("\n".join(describe_run(report, device, amp_enabled)))
    print()
    print("\n".join(format_report(report)))

    output_dir: Path = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON_FILENAME
    csv_path = output_dir / SLICE_CSV_FILENAME
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_slice_csv(csv_path, records, samples)
    print(f"\nDiagnostics saved to '{json_path}' and '{csv_path}'.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
