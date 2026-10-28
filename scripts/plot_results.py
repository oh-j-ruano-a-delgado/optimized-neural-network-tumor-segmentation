"""Render the Phase 4 comparison figures and tables for the baseline and GA-optimized runs.

Figures (in ``<output-dir>/figures``):
    loss_curves.png        training and validation loss per epoch for both runs
    validation_dice.png    validation Dice per epoch with best epochs and the crossing point
    test_metrics.png       per-class, mean and whole-tumor Dice/IoU on the held-out test set
                           (skipped until ``python -m scripts.evaluate_test_set`` has run)

Tables (in ``<output-dir>``), the table view behind the figures:
    convergence_summary.csv     best epoch, best Dice, time to best, epoch reaching baseline best
    training_curves_aligned.csv both histories side by side on a shared epoch index (NaN-padded)

Usage (from the project root; P4 = runs/phase4_baseline_vs_ga):
    python -m scripts.plot_results [--baseline-history P4/01_baseline_manual/history.csv]
        [--optimized-history P4/02_A_ga_optimized/history.csv]
        [--test-metrics P4/03_test_comparison_baseline_vs_A/test_metrics.csv]
        [--output-dir P4/03_test_comparison_baseline_vs_A]
        [--smoothing-window 5] [--formats png pdf] [--dpi 200]
"""

import argparse
from pathlib import Path
from typing import Final, List, Sequence

import pandas as pd
from matplotlib.figure import Figure

from src.utils.visualization import (
    DEFAULT_DPI,
    DEFAULT_SMOOTHING_WINDOW,
    REQUIRED_HISTORY_COLUMNS,
    TrainingRun,
    align_histories,
    build_convergence_table,
    load_history,
    load_test_metrics,
    plot_loss_curves,
    plot_test_metrics,
    plot_validation_dice,
    save_figure,
)

PHASE4_RUNS_DIR: Final[Path] = Path("runs/baseline_vs_ga")
DEFAULT_BASELINE_HISTORY: Final[Path] = PHASE4_RUNS_DIR / "01_baseline_manual" / "history.csv"
DEFAULT_OPTIMIZED_HISTORY: Final[Path] = PHASE4_RUNS_DIR / "02_A_ga_optimized" / "history.csv"
DEFAULT_OUTPUT_DIR: Final[Path] = PHASE4_RUNS_DIR / "03_test_comparison_baseline_vs_A"
FIGURES_DIRNAME: Final[str] = "figures"
CONVERGENCE_FILENAME: Final[str] = "convergence_summary.csv"
ALIGNED_HISTORY_FILENAME: Final[str] = "training_curves_aligned.csv"
ALIGNED_COLUMNS: Final[Sequence[str]] = (
    *REQUIRED_HISTORY_COLUMNS[1:],
    "val_iou_mean",
    "elapsed_seconds",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-history", type=Path, default=DEFAULT_BASELINE_HISTORY)
    parser.add_argument("--optimized-history", type=Path, default=DEFAULT_OPTIMIZED_HISTORY)
    parser.add_argument("--test-metrics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    parser.add_argument("--formats", nargs="+", default=["png"])
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser.parse_args()


def describe_run(run: TrainingRun) -> str:
    return f"{run.style.label}: {len(run.history)} epochs loaded"


def export_figure(
    figure: Figure, output_dir: Path, stem: str, formats: Sequence[str], dpi: int
) -> List[Path]:
    saved_paths = save_figure(figure, output_dir, stem, formats, dpi)
    for path in saved_paths:
        print(f"Saved '{path}'.")
    return saved_paths


def main() -> None:
    arguments = parse_arguments()
    output_dir: Path = arguments.output_dir
    figures_dir = output_dir / FIGURES_DIRNAME
    test_metrics_path: Path = arguments.test_metrics or output_dir / "test_metrics.csv"

    runs = [
        TrainingRun("baseline", load_history(arguments.baseline_history)),
        TrainingRun("optimized", load_history(arguments.optimized_history)),
    ]
    for run in runs:
        print(describe_run(run))

    convergence_table = build_convergence_table(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence_table.to_csv(output_dir / CONVERGENCE_FILENAME, index=False)
    align_histories(runs, ALIGNED_COLUMNS).to_csv(
        output_dir / ALIGNED_HISTORY_FILENAME, index=False
    )
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(convergence_table.to_string(index=False))

    export_figure(
        plot_loss_curves(runs, arguments.smoothing_window),
        figures_dir,
        "loss_curves",
        arguments.formats,
        arguments.dpi,
    )
    export_figure(
        plot_validation_dice(runs, arguments.smoothing_window),
        figures_dir,
        "validation_dice",
        arguments.formats,
        arguments.dpi,
    )
    if test_metrics_path.is_file():
        export_figure(
            plot_test_metrics(load_test_metrics(test_metrics_path)),
            figures_dir,
            "test_metrics",
            arguments.formats,
            arguments.dpi,
        )
    else:
        print(
            f"Skipping the test metric chart: '{test_metrics_path}' not found. "
            "Run 'python -m scripts.evaluate_test_set' first."
        )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
