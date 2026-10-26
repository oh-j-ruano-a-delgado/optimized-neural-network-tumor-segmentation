"""Tests for the comparison plotting helpers, focused on histories of different lengths."""

import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.utils.visualization import (
    TrainingRun,
    align_histories,
    build_convergence_table,
    first_epoch_reaching,
    load_history,
    plot_loss_curves,
    plot_test_metrics,
    plot_validation_dice,
    rolling_mean,
    save_figure,
    summarize_convergence,
)


def build_history(validation_dice: List[float], elapsed_seconds: float = 60.0) -> pd.DataFrame:
    epochs = np.arange(1, len(validation_dice) + 1)
    return pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": 0.5 / epochs,
            "val_loss": 0.6 / epochs,
            "val_dice_mean": validation_dice,
            "val_iou_mean": [score * 0.7 for score in validation_dice],
            "elapsed_seconds": elapsed_seconds,
        }
    )


@pytest.fixture
def runs_of_different_lengths() -> List[TrainingRun]:
    return [
        TrainingRun("baseline", build_history([0.2, 0.4, 0.5])),
        TrainingRun("optimized", build_history([0.3, 0.55, 0.6, 0.7, 0.65])),
    ]


@pytest.fixture
def test_metrics() -> pd.DataFrame:
    regions = ["meningioma", "glioma", "pituitary", "mean", "whole_tumor"]
    rows = [
        {"model": model, "region": region, "dice": dice, "iou": dice * 0.7}
        for model, base_score in (("baseline", 0.6), ("optimized", 0.7))
        for region, dice in zip(regions, np.linspace(base_score, base_score + 0.1, len(regions)))
    ]
    return pd.DataFrame(rows)


def test_align_histories_pads_the_shorter_run(runs_of_different_lengths: List[TrainingRun]) -> None:
    aligned = align_histories(runs_of_different_lengths, ["val_dice_mean", "missing_column"])

    assert aligned["epoch"].tolist() == [1, 2, 3, 4, 5]
    assert aligned["baseline_val_dice_mean"].isna().tolist() == [False, False, False, True, True]
    assert aligned["optimized_val_dice_mean"].notna().all()
    assert aligned["baseline_missing_column"].isna().all()


def test_summarize_convergence_reports_first_best_epoch_and_time() -> None:
    summary = summarize_convergence(build_history([0.1, 0.5, 0.5, 0.3], elapsed_seconds=30.0))

    assert summary.best_epoch == 2
    assert summary.best_value == pytest.approx(0.5)
    assert summary.total_epochs == 4
    assert summary.seconds_to_best == pytest.approx(60.0)
    assert summary.total_seconds == pytest.approx(120.0)


def test_summarize_convergence_without_timing_column() -> None:
    history = build_history([0.1, 0.4]).drop(columns="elapsed_seconds")
    summary = summarize_convergence(history)

    assert summary.seconds_to_best is None
    assert summary.total_seconds is None


def test_minimizing_summary_finds_lowest_loss() -> None:
    summary = summarize_convergence(build_history([0.1, 0.2, 0.3]), "val_loss", maximize=False)
    assert summary.best_epoch == 3


def test_first_epoch_reaching_threshold() -> None:
    history = build_history([0.2, 0.6, 0.4, 0.7])
    assert first_epoch_reaching(history, "val_dice_mean", 0.6) == 2
    assert first_epoch_reaching(history, "val_dice_mean", 0.9) is None


def test_rolling_mean_window_of_one_is_identity() -> None:
    values = pd.Series([0.1, 0.9, 0.2])
    assert rolling_mean(values, 1).tolist() == values.tolist()
    assert rolling_mean(values, 3).iloc[1] == pytest.approx(0.4)


def test_convergence_table_tracks_epoch_reaching_reference_best(
    runs_of_different_lengths: List[TrainingRun],
) -> None:
    table = build_convergence_table(runs_of_different_lengths).set_index("model")

    assert table.at["baseline", "epochs_trained"] == 3
    assert table.at["optimized", "epochs_trained"] == 5
    assert table.at["optimized", "best_epoch"] == 4
    assert table.at["optimized", "best_validation_dice_within_baseline_epochs"] == pytest.approx(
        0.6
    )
    assert table.at["optimized", "epoch_reaching_baseline_best"] == 2
    assert table.at["optimized", "minutes_to_baseline_best"] == pytest.approx(2.0)


def test_load_history_rejects_missing_columns(tmp_path: Path) -> None:
    history_path = tmp_path / "history.csv"
    pd.DataFrame({"epoch": [1], "train_loss": [0.5]}).to_csv(history_path, index=False)
    with pytest.raises(ValueError, match="val_loss"):
        load_history(history_path)


def test_unknown_model_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        TrainingRun("mystery", build_history([0.1])).style


def test_figures_render_for_runs_of_different_lengths(
    tmp_path: Path, runs_of_different_lengths: List[TrainingRun], test_metrics: pd.DataFrame
) -> None:
    figures = {
        "loss_curves": plot_loss_curves(runs_of_different_lengths, smoothing_window=3),
        "validation_dice": plot_validation_dice(runs_of_different_lengths, smoothing_window=3),
        "test_metrics": plot_test_metrics(test_metrics),
    }

    assert len(figures["loss_curves"].axes) == 2
    assert len(figures["validation_dice"].axes) == 1
    assert len(figures["test_metrics"].axes) == 2
    validation_axis = figures["validation_dice"].axes[0]
    assert validation_axis.get_xlim()[1] >= 5
    for stem, figure in figures.items():
        (saved_path,) = save_figure(figure, tmp_path, stem, dpi=40)
        assert saved_path.is_file() and saved_path.stat().st_size > 0


def test_test_metric_chart_tolerates_missing_scores(test_metrics: pd.DataFrame) -> None:
    incomplete = test_metrics[
        ~((test_metrics["model"] == "optimized") & (test_metrics["region"] == "glioma"))
    ]
    figure = plot_test_metrics(incomplete)
    assert not math.isnan(figure.axes[0].get_ylim()[1])
