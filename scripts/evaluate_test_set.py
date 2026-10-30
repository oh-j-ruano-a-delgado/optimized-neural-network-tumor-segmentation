"""Evaluate two segmentation checkpoints against each other on the held-out Testing split.

Neither model saw the Testing split during training, checkpoint selection or the genetic
search, so this is the unbiased comparison (Phase 4 onwards). Each checkpoint is rebuilt from its
own stored architecture, either the custom U-Net or the SMP U-Net with a pretrained encoder (see
``src/models/factory.py``), and preprocessed with the deterministic evaluation transform at the
image size it was trained with. Dice and IoU are accumulated over every test pixel (the same global
aggregation used for validation) and reported per tumor class, as their macro mean over the
three classes (the validation metric), and for the whole tumor, where every tumor class is
merged into one foreground label so localization quality can be told apart from tumor-type
confusion.

The two checkpoints are reported under ``--reference-name`` and ``--challenger-name`` (default
"baseline" and "optimized", the Phase 4 pair that ``scripts/plot_results.py`` styles). Later
comparisons, such as the GA-optimized Run A against the Phase 6 Run E or Run E against the
Phase 7 transfer-learning Run F, pass their own names so the tables and the JSON never mislabel a
run; the names must differ because they key the JSON. The line printed for each checkpoint names
its architecture: filters and depth for the custom U-Net, the encoder for the SMP U-Net.

Outputs (in ``--output-dir``):
    test_metrics.csv        tidy table (model, region, dice, iou) read by scripts/plot_results.py
    test_metrics.json       the same scores plus checkpoint metadata and test-set composition
    test_slices.csv         one row per slice (model, sample, class, dice, precision, recall, ...)
    test_slice_metrics.json per-model detection rate, missed tumors and per-slice Dice summary

Usage (from the project root; P4 = runs/baseline_vs_ga):
    python -m scripts.evaluate_test_set [--baseline-checkpoint P4/01_baseline_manual/best_model.pth]
        [--optimized-checkpoint P4/02_A_ga_optimized/best_model.pth]
        [--device auto|cpu|cuda] [--amp] [--batch-size 8]
        [--output-dir P4/03_test_comparison_baseline_vs_A]
        [--reference-name baseline] [--challenger-name optimized]
"""

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
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
    TUMOR_CLASS_NAMES,
)
from src.dataset.dataloader import (
    BrainTumorSegmentationDataset,
    SegmentationBatch,
    SegmentationSample,
    build_dataloader,
    discover_samples,
)
from src.dataset.transforms import build_evaluation_transform
from src.models.factory import describe_model
from src.utils.checkpoint import (
    TrainingCheckpoint,
    build_model_from_checkpoint,
    load_training_checkpoint,
)
from src.utils.metrics import SegmentationMetrics, SegmentationReport
from src.utils.slice_metrics import (
    SliceScore,
    SliceSummary,
    score_slice,
    summarize_slices,
)
from src.utils.training import resolve_amp_mode, resolve_device, transfer_batch

PHASE4_RUNS_DIR: Final[Path] = Path("runs/baseline_vs_ga")
DEFAULT_BASELINE_CHECKPOINT: Final[Path] = PHASE4_RUNS_DIR / "01_baseline_manual" / "best_model.pth"
DEFAULT_OPTIMIZED_CHECKPOINT: Final[Path] = PHASE4_RUNS_DIR / "02_A_ga_optimized" / "best_model.pth"
DEFAULT_OUTPUT_DIR: Final[Path] = PHASE4_RUNS_DIR / "03_test_comparison_baseline_vs_A"
DEFAULT_BATCH_SIZE: Final[int] = 8
DEFAULT_SEED: Final[int] = 42
METRICS_CSV_FILENAME: Final[str] = "test_metrics.csv"
METRICS_JSON_FILENAME: Final[str] = "test_metrics.json"
SLICES_CSV_FILENAME: Final[str] = "test_slices.csv"
SLICE_METRICS_JSON_FILENAME: Final[str] = "test_slice_metrics.json"
MEAN_REGION: Final[str] = "mean"
WHOLE_TUMOR_REGION: Final[str] = "whole_tumor"
WHOLE_TUMOR_CLASSES: Final[int] = 2
CSV_FIELDNAMES: Final[Tuple[str, ...]] = ("model", "region", "dice", "iou")
SLICE_CSV_FIELDNAMES: Final[Tuple[str, ...]] = (
    "model",
    "sample",
    "class",
    "dice",
    "precision",
    "recall",
    "tumor_pixels",
    "predicted_pixels",
    "detected",
)
REFERENCE_MODEL: Final[str] = "baseline"
CHALLENGER_MODEL: Final[str] = "optimized"
MINIMUM_SCORE_COLUMN_WIDTH: Final[int] = 10


@dataclass(frozen=True)
class RegionScore:
    region: str
    dice: float
    iou: float


@dataclass(frozen=True)
class ModelEvaluation:
    name: str
    checkpoint: TrainingCheckpoint
    scores: Tuple[RegionScore, ...]
    slice_scores: Tuple[SliceScore, ...]
    evaluation_seconds: float

    def score_for(self, region: str) -> RegionScore:
        return next(score for score in self.scores if score.region == region)

    @property
    def slice_summary(self) -> SliceSummary:
        return summarize_slices(self.slice_scores)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--optimized-checkpoint", type=Path, default=DEFAULT_OPTIMIZED_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-name", default=REFERENCE_MODEL)
    parser.add_argument("--challenger-name", default=CHALLENGER_MODEL)
    return parser.parse_args()


def validate_model_names(reference_name: str, challenger_name: str) -> None:
    if reference_name == challenger_name:
        raise ValueError(
            f"--reference-name and --challenger-name must differ, both are '{reference_name}'."
        )


def build_test_loader(
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


def to_whole_tumor_labels(class_indices: Tensor) -> Tensor:
    return (class_indices != BACKGROUND_LABEL).long()


def accumulate_test_metrics(
    model: nn.Module,
    loader: DataLoader[SegmentationBatch],
    samples: Sequence[SegmentationSample],
    model_name: str,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[SegmentationReport, SegmentationReport, Tuple[SliceScore, ...]]:
    class_metrics = SegmentationMetrics(num_classes=NUM_CLASSES)
    whole_tumor_metrics = SegmentationMetrics(num_classes=WHOLE_TUMOR_CLASSES)
    slice_scores: List[SliceScore] = []
    next_sample_index = 0
    model.eval()
    with torch.inference_mode():
        for images, masks in loader:
            images, masks = transfer_batch(images, masks, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predictions = model(images).argmax(dim=1)
            class_metrics.update(predictions, masks)
            whole_tumor_metrics.update(
                to_whole_tumor_labels(predictions), to_whole_tumor_labels(masks)
            )
            batch_predictions = to_whole_tumor_labels(predictions).cpu().numpy()
            batch_masks = to_whole_tumor_labels(masks).cpu().numpy()
            for prediction, mask in zip(batch_predictions, batch_masks):
                sample = samples[next_sample_index]
                slice_scores.append(
                    score_slice(
                        prediction,
                        mask,
                        model_name,
                        sample.image_path.stem,
                        TUMOR_CLASS_NAMES[sample.class_label],
                    )
                )
                next_sample_index += 1
    return class_metrics.compute(), whole_tumor_metrics.compute(), tuple(slice_scores)


def build_region_scores(
    class_report: SegmentationReport, whole_tumor_report: SegmentationReport
) -> Tuple[RegionScore, ...]:
    class_scores = [
        RegionScore(TUMOR_CLASS_NAMES[class_index], float(dice), float(iou))
        for class_index, dice, iou in zip(
            class_report.class_indices,
            class_report.dice_by_class.tolist(),
            class_report.iou_by_class.tolist(),
        )
    ]
    return (
        *class_scores,
        RegionScore(MEAN_REGION, class_report.mean_dice, class_report.mean_iou),
        RegionScore(
            WHOLE_TUMOR_REGION,
            float(whole_tumor_report.dice_by_class[0]),
            float(whole_tumor_report.iou_by_class[0]),
        ),
    )


def evaluate_checkpoint(
    name: str,
    checkpoint_path: Path,
    samples: Sequence[SegmentationSample],
    arguments: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
) -> ModelEvaluation:
    checkpoint = load_training_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint, device)
    loader = build_test_loader(
        samples, checkpoint.image_size, arguments.batch_size, arguments.num_workers, arguments.seed
    )
    start_time = time.perf_counter()
    class_report, whole_tumor_report, slice_scores = accumulate_test_metrics(
        model, loader, samples, name, device, amp_enabled
    )
    if len(slice_scores) != len(samples):
        raise ValueError(
            f"Scored {len(slice_scores)} slices but the test set has {len(samples)}."
        )
    return ModelEvaluation(
        name=name,
        checkpoint=checkpoint,
        scores=build_region_scores(class_report, whole_tumor_report),
        slice_scores=slice_scores,
        evaluation_seconds=time.perf_counter() - start_time,
    )


def finite_or_none(value: float) -> Optional[float]:
    return round(value, 6) if math.isfinite(value) else None


def count_samples_per_class(samples: Sequence[SegmentationSample]) -> Dict[str, int]:
    counts = Counter(sample.class_label for sample in samples)
    return {TUMOR_CLASS_NAMES[label]: counts[label] for label in sorted(TUMOR_CLASS_NAMES)}


def write_metrics_csv(path: Path, evaluations: Sequence[ModelEvaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            {
                "model": evaluation.name,
                "region": score.region,
                "dice": finite_or_none(score.dice),
                "iou": finite_or_none(score.iou),
            }
            for evaluation in evaluations
            for score in evaluation.scores
        )


def serialize_evaluation(evaluation: ModelEvaluation) -> Dict[str, Any]:
    checkpoint = evaluation.checkpoint
    return {
        "checkpoint": checkpoint.path.as_posix(),
        "best_epoch": checkpoint.epoch,
        "validation_dice": finite_or_none(checkpoint.validation_dice),
        "hyperparameters": checkpoint.hyperparameters,
        "image_size": checkpoint.image_size,
        "test": {
            score.region: {"dice": finite_or_none(score.dice), "iou": finite_or_none(score.iou)}
            for score in evaluation.scores
        },
    }


def write_metrics_json(
    path: Path,
    evaluations: Sequence[ModelEvaluation],
    samples_per_class: Mapping[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "test_samples": sum(samples_per_class.values()),
        "test_samples_per_class": dict(samples_per_class),
        "models": {
            evaluation.name: serialize_evaluation(evaluation) for evaluation in evaluations
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_slices_csv(path: Path, evaluations: Sequence[ModelEvaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SLICE_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            {
                "model": score.model,
                "sample": score.sample_name,
                "class": score.class_name,
                "dice": finite_or_none(score.dice),
                "precision": finite_or_none(score.precision),
                "recall": finite_or_none(score.recall),
                "tumor_pixels": score.tumor_pixels,
                "predicted_pixels": score.predicted_pixels,
                "detected": score.detected,
            }
            for evaluation in evaluations
            for score in evaluation.slice_scores
        )


def write_slice_metrics_json(
    path: Path,
    evaluations: Sequence[ModelEvaluation],
    samples_per_class: Mapping[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "test_samples": sum(samples_per_class.values()),
        "test_samples_per_class": dict(samples_per_class),
        "models": {
            evaluation.name: evaluation.slice_summary.to_dict() for evaluation in evaluations
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_slice_summary(evaluation: ModelEvaluation) -> str:
    summary = evaluation.slice_summary
    detection_by_class = " · ".join(
        f"{name} {rate:.1%}" for name, rate in summary.detection_rate_by_class.items()
    )
    return (
        f"{evaluation.name:<10} | detection {summary.detection_rate:.1%} | "
        f"empty predictions {summary.predicted_empty_slices}/{summary.slices} | "
        f"missed tumors {summary.missed_tumor_slices} | "
        f"false positives {summary.false_positive_slices} | "
        f"Dice mean {summary.dice_mean:.3f} median {summary.dice_median:.3f} | "
        f"pooled precision {summary.precision:.3f} recall {summary.recall:.3f} | "
        f"by class: {detection_by_class}"
    )


def format_comparison_table(reference: ModelEvaluation, challenger: ModelEvaluation) -> List[str]:
    width = max(MINIMUM_SCORE_COLUMN_WIDTH, len(reference.name), len(challenger.name)) + 7
    header = (
        f"{'Region':<13}{'Dice ' + reference.name:>{width}}{'Dice ' + challenger.name:>{width}}"
        f"{'Delta':>9}{'IoU ' + reference.name:>{width}}{'IoU ' + challenger.name:>{width}}"
        f"{'Delta':>9}"
    )
    lines = [header, "-" * len(header)]
    for reference_score in reference.scores:
        challenger_score = challenger.score_for(reference_score.region)
        lines.append(
            f"{reference_score.region:<13}"
            f"{reference_score.dice:>{width}.4f}{challenger_score.dice:>{width}.4f}"
            f"{challenger_score.dice - reference_score.dice:>+9.4f}"
            f"{reference_score.iou:>{width}.4f}{challenger_score.iou:>{width}.4f}"
            f"{challenger_score.iou - reference_score.iou:>+9.4f}"
        )
    return lines


def describe_checkpoint(evaluation: ModelEvaluation) -> str:
    checkpoint = evaluation.checkpoint
    hyperparameters = checkpoint.hyperparameters
    return (
        f"{evaluation.name:<16} | epoch {checkpoint.epoch:>3} | "
        f"validation Dice {checkpoint.validation_dice:.4f} | "
        f"lr {hyperparameters.get('learning_rate', float('nan')):.3g} | "
        f"wd {hyperparameters.get('weight_decay', float('nan')):.3g} | "
        f"{describe_model(checkpoint.model_settings)} | "
        f"evaluated in {evaluation.evaluation_seconds:.1f}s"
    )


def main() -> None:
    arguments = parse_arguments()
    validate_model_names(arguments.reference_name, arguments.challenger_name)
    device = resolve_device(arguments.device)
    amp_enabled = resolve_amp_mode(arguments.amp, device)
    samples = discover_samples(arguments.data_dir / TESTING_SPLIT)
    samples_per_class = count_samples_per_class(samples)
    print(
        f"Device: {device} | AMP: {'on' if amp_enabled else 'off'} | "
        f"test set {len(samples)} slices ("
        + ", ".join(f"{name} {count}" for name, count in samples_per_class.items())
        + ")"
    )

    evaluations = [
        evaluate_checkpoint(name, checkpoint_path, samples, arguments, device, amp_enabled)
        for name, checkpoint_path in (
            (arguments.reference_name, arguments.baseline_checkpoint),
            (arguments.challenger_name, arguments.optimized_checkpoint),
        )
    ]
    for evaluation in evaluations:
        print(describe_checkpoint(evaluation))

    reference, challenger = evaluations
    print()
    print("\n".join(format_comparison_table(reference, challenger)))

    output_dir: Path = arguments.output_dir
    csv_path = output_dir / METRICS_CSV_FILENAME
    json_path = output_dir / METRICS_JSON_FILENAME
    slices_csv_path = output_dir / SLICES_CSV_FILENAME
    slice_metrics_path = output_dir / SLICE_METRICS_JSON_FILENAME
    write_metrics_csv(csv_path, evaluations)
    write_metrics_json(json_path, evaluations, samples_per_class)
    write_slices_csv(slices_csv_path, evaluations)
    write_slice_metrics_json(slice_metrics_path, evaluations, samples_per_class)

    print("\nPer-slice whole-tumor metrics:")
    for evaluation in evaluations:
        print(format_slice_summary(evaluation))
    print(
        f"\nTest metrics saved to '{csv_path}' and '{json_path}'.\n"
        f"Per-slice metrics saved to '{slices_csv_path}' and '{slice_metrics_path}'."
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
