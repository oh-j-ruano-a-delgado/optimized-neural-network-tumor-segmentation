"""Presentation-grade comparison plots for the baseline and GA-optimized U-Net runs.

Every figure shares one dark, high-contrast theme applied through ``matplotlib.rc_context``,
so importing this module never mutates global matplotlib state, and figures are built on
``matplotlib.figure.Figure`` directly so no GUI backend is needed. Model identity is fixed by
model name (electric blue for the baseline, neon pink for the GA-optimized run); the pair was
checked against the dark surface for lightness, chroma, contrast and color-vision-deficiency
separation, and is always backed by a legend plus direct labels so it never relies on color
alone. Text always uses neutral ink tokens, never the series color.

Training histories are treated as independent series: each curve is drawn against its own
``epoch`` column, axis limits span the longest run, an end marker shows where each run stops,
and summaries such as the best epoch or the cumulative wall-clock time to reach it are computed
per run. Runs of different lengths (the 40-epoch baseline next to the 100-epoch optimized run)
therefore need no padding or truncation; ``align_histories`` provides the side-by-side table
view, where shorter runs are padded with NaN.

Noisy per-epoch values are drawn as thin translucent traces beneath a thick, glowing centered
rolling mean, while markers, annotations and summaries always refer to the raw values.

The loss axis is labelled generically ("Loss" unless ``loss_label`` says otherwise) because the
Phase 5 runs train with different criteria (Dice-Focal, Focal Tversky-Focal, Dice-CE) whose
values are not on a common scale; runs trained with different losses must be compared by
validation Dice, not by loss.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Final, Iterator, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager, patheffects
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

EPOCH_COLUMN: Final[str] = "epoch"
DEFAULT_LOSS_LABEL: Final[str] = "Loss"
TRAIN_LOSS_COLUMN: Final[str] = "train_loss"
VALIDATION_LOSS_COLUMN: Final[str] = "val_loss"
VALIDATION_DICE_COLUMN: Final[str] = "val_dice_mean"
ELAPSED_COLUMN: Final[str] = "elapsed_seconds"
REQUIRED_HISTORY_COLUMNS: Final[Tuple[str, ...]] = (
    EPOCH_COLUMN,
    TRAIN_LOSS_COLUMN,
    VALIDATION_LOSS_COLUMN,
    VALIDATION_DICE_COLUMN,
)
REQUIRED_TEST_METRIC_COLUMNS: Final[Tuple[str, ...]] = ("model", "region", "dice", "iou")

SURFACE_COLOR: Final[str] = "#0B0F1A"
RAISED_SURFACE_COLOR: Final[str] = "#151B2B"
GRID_COLOR: Final[str] = "#1B2233"
AXIS_COLOR: Final[str] = "#2E3850"
TEXT_PRIMARY_COLOR: Final[str] = "#EEF1F7"
TEXT_SECONDARY_COLOR: Final[str] = "#A9B2C5"
TEXT_MUTED_COLOR: Final[str] = "#727D95"
BASELINE_COLOR: Final[str] = "#2F80FF"
OPTIMIZED_COLOR: Final[str] = "#FF2E9A"
FONT_CANDIDATES: Final[Tuple[str, ...]] = (
    "Inter",
    "Segoe UI",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "DejaVu Sans",
)
FALLBACK_FONT: Final[str] = "DejaVu Sans"

LINE_WIDTH: Final[float] = 3.2
RAW_TRACE_WIDTH: Final[float] = 1.3
RAW_TRACE_ALPHA: Final[float] = 0.42
END_MARKER_SIZE: Final[float] = 9.0
HIGHLIGHT_MARKER_SIZE: Final[float] = 12.0
HIGHLIGHT_HALO_SIZE: Final[float] = 24.0
MARKER_RING_WIDTH: Final[float] = 2.0
BAR_WIDTH: Final[float] = 0.36
BAR_GAP: Final[float] = 0.03
AGGREGATE_GROUP_OFFSET: Final[float] = 0.6
DEFAULT_SMOOTHING_WINDOW: Final[int] = 5
DEFAULT_DPI: Final[int] = 200
HEADER_HEIGHT_INCHES: Final[float] = 1.3
FOOTER_HEIGHT_INCHES: Final[float] = 0.7
SIDE_MARGIN_INCHES: Final[float] = 0.3
SECONDS_PER_MINUTE: Final[float] = 60.0


@dataclass(frozen=True)
class ModelStyle:
    label: str
    short_label: str
    color: str


MODEL_STYLES: Final[Mapping[str, ModelStyle]] = MappingProxyType(
    {
        "baseline": ModelStyle("Baseline (manual)", "Baseline", BASELINE_COLOR),
        "optimized": ModelStyle("Optimized (GA)", "Optimized", OPTIMIZED_COLOR),
    }
)
MODEL_ORDER: Final[Tuple[str, ...]] = tuple(MODEL_STYLES)
REFERENCE_MODEL: Final[str] = "baseline"
REGION_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "meningioma": "Meningioma",
        "glioma": "Glioma",
        "pituitary": "Pituitary",
        "mean": "Mean\n(3 classes)",
        "whole_tumor": "Whole\ntumor",
    }
)
AGGREGATE_REGIONS: Final[Tuple[str, ...]] = ("mean", "whole_tumor")


@dataclass(frozen=True)
class TrainingRun:
    name: str
    history: pd.DataFrame

    @property
    def style(self) -> ModelStyle:
        return resolve_model_style(self.name)


@dataclass(frozen=True)
class ConvergenceSummary:
    total_epochs: int
    best_epoch: int
    best_value: float
    seconds_to_best: Optional[float]
    total_seconds: Optional[float]


def resolve_model_style(name: str) -> ModelStyle:
    if name not in MODEL_STYLES:
        raise ValueError(f"Unknown model '{name}'; expected one of {list(MODEL_STYLES)}.")
    return MODEL_STYLES[name]


def find_run(runs: Sequence[TrainingRun], name: str) -> Optional[TrainingRun]:
    return next((run for run in runs if run.name == name), None)


def load_history(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"History file '{path}' does not exist.")
    history = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_HISTORY_COLUMNS if column not in history]
    if missing_columns:
        raise ValueError(f"History '{path}' is missing the columns {missing_columns}.")
    if history.empty:
        raise ValueError(f"History '{path}' has no epochs.")
    return history.sort_values(EPOCH_COLUMN).reset_index(drop=True)


def load_test_metrics(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Test metrics file '{path}' does not exist.")
    test_metrics = pd.read_csv(path)
    missing_columns = [
        column for column in REQUIRED_TEST_METRIC_COLUMNS if column not in test_metrics
    ]
    if missing_columns:
        raise ValueError(f"Test metrics '{path}' are missing the columns {missing_columns}.")
    return test_metrics


def rolling_mean(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return values.astype(float)
    return values.rolling(window, center=True, min_periods=1).mean()


def cumulative_training_seconds(history: pd.DataFrame) -> Optional[pd.Series]:
    if ELAPSED_COLUMN not in history:
        return None
    return history[ELAPSED_COLUMN].fillna(0.0).cumsum()


def summarize_convergence(
    history: pd.DataFrame, metric_column: str = VALIDATION_DICE_COLUMN, maximize: bool = True
) -> ConvergenceSummary:
    metric = history[metric_column]
    if metric.notna().sum() == 0:
        raise ValueError(f"Column '{metric_column}' has no finite values.")
    best_index = metric.idxmax() if maximize else metric.idxmin()
    cumulative_seconds = cumulative_training_seconds(history)
    return ConvergenceSummary(
        total_epochs=int(history[EPOCH_COLUMN].max()),
        best_epoch=int(history.at[best_index, EPOCH_COLUMN]),
        best_value=float(metric.at[best_index]),
        seconds_to_best=(
            None if cumulative_seconds is None else float(cumulative_seconds.at[best_index])
        ),
        total_seconds=None if cumulative_seconds is None else float(cumulative_seconds.iloc[-1]),
    )


def first_epoch_reaching(
    history: pd.DataFrame, metric_column: str, threshold: float
) -> Optional[int]:
    reached_epochs = history.loc[history[metric_column] >= threshold, EPOCH_COLUMN]
    return None if reached_epochs.empty else int(reached_epochs.iloc[0])


def seconds_until_epoch(history: pd.DataFrame, epoch: int) -> Optional[float]:
    cumulative_seconds = cumulative_training_seconds(history)
    if cumulative_seconds is None:
        return None
    return float(cumulative_seconds[history[EPOCH_COLUMN] <= epoch].iloc[-1])


def align_histories(runs: Sequence[TrainingRun], columns: Sequence[str]) -> pd.DataFrame:
    frames = [
        run.history.set_index(EPOCH_COLUMN)
        .reindex(columns=list(columns))
        .add_prefix(f"{run.name}_")
        for run in runs
    ]
    return (
        pd.concat(frames, axis=1, join="outer").sort_index().rename_axis(EPOCH_COLUMN).reset_index()
    )


def best_value_within_epochs(
    history: pd.DataFrame, metric_column: str, max_epoch: int
) -> Optional[float]:
    within_budget = history.loc[history[EPOCH_COLUMN] <= max_epoch, metric_column]
    return None if within_budget.notna().sum() == 0 else float(within_budget.max())


def to_minutes(seconds: Optional[float]) -> Optional[float]:
    return None if seconds is None else round(seconds / SECONDS_PER_MINUTE, 2)


def build_convergence_table(
    runs: Sequence[TrainingRun], reference_name: str = REFERENCE_MODEL
) -> pd.DataFrame:
    reference = find_run(runs, reference_name)
    reference_summary = None if reference is None else summarize_convergence(reference.history)
    rows: List[Dict[str, Any]] = []
    for run in runs:
        dice_summary = summarize_convergence(run.history)
        loss_summary = summarize_convergence(run.history, VALIDATION_LOSS_COLUMN, maximize=False)
        epoch_reaching_reference = (
            None
            if reference_summary is None
            else first_epoch_reaching(
                run.history, VALIDATION_DICE_COLUMN, reference_summary.best_value
            )
        )
        best_within_reference_budget = (
            None
            if reference_summary is None
            else best_value_within_epochs(
                run.history, VALIDATION_DICE_COLUMN, reference_summary.total_epochs
            )
        )
        rows.append(
            {
                "model": run.name,
                "epochs_trained": dice_summary.total_epochs,
                "best_epoch": dice_summary.best_epoch,
                "best_validation_dice": round(dice_summary.best_value, 6),
                "minutes_to_best": to_minutes(dice_summary.seconds_to_best),
                "total_training_minutes": to_minutes(dice_summary.total_seconds),
                "min_validation_loss": round(loss_summary.best_value, 6),
                "min_validation_loss_epoch": loss_summary.best_epoch,
                f"best_validation_dice_within_{reference_name}_epochs": (
                    None
                    if best_within_reference_budget is None
                    else round(best_within_reference_budget, 6)
                ),
                f"epoch_reaching_{reference_name}_best": epoch_reaching_reference,
                f"minutes_to_{reference_name}_best": (
                    None
                    if epoch_reaching_reference is None
                    else to_minutes(seconds_until_epoch(run.history, epoch_reaching_reference))
                ),
            }
        )
    return pd.DataFrame(rows)


def resolve_font_family(candidates: Sequence[str] = FONT_CANDIDATES) -> str:
    available_families = {font_entry.name for font_entry in font_manager.fontManager.ttflist}
    return next((family for family in candidates if family in available_families), FALLBACK_FONT)


def build_theme() -> Dict[str, Any]:
    return {
        "figure.facecolor": SURFACE_COLOR,
        "figure.edgecolor": SURFACE_COLOR,
        "savefig.facecolor": SURFACE_COLOR,
        "savefig.edgecolor": SURFACE_COLOR,
        "axes.facecolor": SURFACE_COLOR,
        "axes.edgecolor": AXIS_COLOR,
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.labelcolor": TEXT_SECONDARY_COLOR,
        "axes.labelsize": 12.5,
        "axes.labelpad": 10.0,
        "axes.titlecolor": TEXT_PRIMARY_COLOR,
        "axes.titlesize": 15.0,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14.0,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.9,
        "grid.linestyle": "-",
        "grid.alpha": 1.0,
        "xtick.color": AXIS_COLOR,
        "ytick.color": AXIS_COLOR,
        "xtick.labelcolor": TEXT_SECONDARY_COLOR,
        "ytick.labelcolor": TEXT_SECONDARY_COLOR,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "xtick.major.size": 0.0,
        "ytick.major.size": 0.0,
        "xtick.major.pad": 9.0,
        "ytick.major.pad": 9.0,
        "font.family": [resolve_font_family()],
        "font.size": 12.0,
        "text.color": TEXT_PRIMARY_COLOR,
        "legend.frameon": False,
        "legend.fontsize": 12.5,
        "legend.labelcolor": TEXT_PRIMARY_COLOR,
        "legend.handlelength": 2.4,
        "legend.columnspacing": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }


@contextmanager
def neon_theme() -> Iterator[None]:
    with matplotlib.rc_context(build_theme()):
        yield


def create_figure(
    figsize: Tuple[float, float], columns: int, share_y: bool = False
) -> Tuple[Figure, List[Axes]]:
    figure = Figure(figsize=figsize, layout="constrained")
    width, height = figsize
    figure.get_layout_engine().set(
        rect=(
            SIDE_MARGIN_INCHES / width,
            FOOTER_HEIGHT_INCHES / height,
            1.0 - 2.0 * SIDE_MARGIN_INCHES / width,
            1.0 - (HEADER_HEIGHT_INCHES + FOOTER_HEIGHT_INCHES) / height,
        ),
        wspace=0.04,
        w_pad=0.12,
    )
    axes = figure.subplots(1, columns, sharey=share_y, squeeze=False)[0].tolist()
    return figure, axes


def add_header(figure: Figure, title: str, subtitle: str) -> None:
    width, height = figure.get_size_inches()
    left = SIDE_MARGIN_INCHES / width
    figure.text(
        left,
        1.0 - 0.3 / height,
        title,
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=TEXT_PRIMARY_COLOR,
    )
    figure.text(
        left,
        1.0 - 0.82 / height,
        subtitle,
        ha="left",
        va="top",
        fontsize=12.5,
        color=TEXT_SECONDARY_COLOR,
    )


def add_footnote(figure: Figure, text: str) -> None:
    width, height = figure.get_size_inches()
    figure.text(
        SIDE_MARGIN_INCHES / width,
        0.16 / height,
        text,
        ha="left",
        va="bottom",
        fontsize=10,
        color=TEXT_MUTED_COLOR,
    )


def add_figure_legend(figure: Figure, handles: Sequence[Artist]) -> None:
    width, height = figure.get_size_inches()
    figure.legend(
        handles=list(handles),
        loc="upper right",
        bbox_to_anchor=(1.0 - SIDE_MARGIN_INCHES / width, 1.0 - 0.22 / height),
        ncol=len(handles),
        frameon=False,
    )


def build_glow_effects(color: str, line_width: float) -> List[AbstractPathEffect]:
    return [
        patheffects.Stroke(linewidth=line_width * 3.6, foreground=color, alpha=0.07),
        patheffects.Stroke(linewidth=line_width * 2.2, foreground=color, alpha=0.16),
        patheffects.Normal(),
    ]


def build_line_handles(runs: Sequence[TrainingRun]) -> List[Line2D]:
    return [
        Line2D(
            [],
            [],
            color=run.style.color,
            linewidth=LINE_WIDTH,
            label=run.style.label,
            path_effects=build_glow_effects(run.style.color, LINE_WIDTH),
        )
        for run in runs
    ]


def build_patch_handles(model_names: Sequence[str]) -> List[Patch]:
    return [
        Patch(facecolor=resolve_model_style(name).color, label=resolve_model_style(name).label)
        for name in model_names
    ]


def draw_point_marker(
    axis: Axes, x: float, y: float, color: str, size: float = END_MARKER_SIZE
) -> None:
    axis.plot(
        [x],
        [y],
        linestyle="none",
        marker="o",
        markersize=size,
        markerfacecolor=color,
        markeredgecolor=SURFACE_COLOR,
        markeredgewidth=MARKER_RING_WIDTH,
        zorder=6,
    )


def draw_highlight_marker(axis: Axes, x: float, y: float, color: str) -> None:
    axis.plot(
        [x],
        [y],
        linestyle="none",
        marker="o",
        markersize=HIGHLIGHT_HALO_SIZE,
        markerfacecolor=color,
        markeredgecolor="none",
        alpha=0.18,
        zorder=6,
    )
    draw_point_marker(axis, x, y, color, HIGHLIGHT_MARKER_SIZE)


def annotate_point(
    axis: Axes,
    x: float,
    y: float,
    text: str,
    offset_points: Tuple[float, float],
    horizontal_alignment: str = "center",
) -> None:
    axis.annotate(
        text,
        xy=(x, y),
        xytext=offset_points,
        textcoords="offset points",
        ha=horizontal_alignment,
        va="center",
        fontsize=11.5,
        fontweight="semibold",
        color=TEXT_PRIMARY_COLOR,
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.5",
            "facecolor": RAISED_SURFACE_COLOR,
            "edgecolor": AXIS_COLOR,
            "linewidth": 0.9,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": TEXT_MUTED_COLOR,
            "linewidth": 1.0,
            "shrinkA": 0.0,
            "shrinkB": 9.0,
        },
        zorder=7,
    )


def plot_metric_trace(
    axis: Axes, run: TrainingRun, column: str, smoothing_window: int
) -> pd.Series:
    epochs = run.history[EPOCH_COLUMN]
    values = run.history[column]
    color = run.style.color
    axis.plot(
        epochs, values, color=color, linewidth=RAW_TRACE_WIDTH, alpha=RAW_TRACE_ALPHA, zorder=2
    )
    smoothed = rolling_mean(values, smoothing_window)
    axis.plot(
        epochs,
        smoothed,
        color=color,
        linewidth=LINE_WIDTH,
        zorder=3,
        path_effects=build_glow_effects(color, LINE_WIDTH),
    )
    last_index = smoothed.last_valid_index()
    if last_index is not None:
        draw_point_marker(axis, epochs.at[last_index], smoothed.at[last_index], color)
    return smoothed


def longest_run_epochs(runs: Sequence[TrainingRun]) -> int:
    return max(int(run.history[EPOCH_COLUMN].max()) for run in runs)


def style_epoch_axis(axis: Axes, runs: Sequence[TrainingRun], title: Optional[str]) -> None:
    max_epoch = longest_run_epochs(runs)
    axis.set_xlim(0, max_epoch + max(1.0, 0.03 * max_epoch))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10], integer=True))
    axis.grid(axis="x", visible=False)
    axis.set_xlabel("Epoch")
    axis.tick_params(axis="both", length=0)
    if title is not None:
        axis.set_title(title)


def describe_smoothing(smoothing_window: int) -> str:
    if smoothing_window <= 1:
        return "Lines show raw per-epoch values"
    return (
        f"Thin traces: raw per-epoch values  ·  thick lines: {smoothing_window}-epoch centered "
        "rolling mean  ·  markers and labels use raw values\n"
        "Small dots mark the last epoch of each run, so runs of different lengths end where "
        "their training stopped"
    )


def format_epoch_label(prefix: str, value: float, epoch: int, seconds: Optional[float]) -> str:
    time_suffix = "" if seconds is None else f" · {seconds / SECONDS_PER_MINUTE:.0f} min"
    return f"{prefix} {value:.4f} · epoch {epoch}{time_suffix}"


def plot_loss_curves(
    runs: Sequence[TrainingRun],
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    loss_label: str = DEFAULT_LOSS_LABEL,
) -> Figure:
    with neon_theme():
        figure, (train_axis, validation_axis) = create_figure((15.0, 6.8), columns=2, share_y=True)
        for axis, column, title in (
            (train_axis, TRAIN_LOSS_COLUMN, "Training loss"),
            (validation_axis, VALIDATION_LOSS_COLUMN, "Validation loss"),
        ):
            for run in runs:
                plot_metric_trace(axis, run, column, smoothing_window)
            style_epoch_axis(axis, runs, title)

        highest_loss = max(
            float(run.history[[TRAIN_LOSS_COLUMN, VALIDATION_LOSS_COLUMN]].max().max())
            for run in runs
        )
        train_axis.set_ylim(0.0, highest_loss * 1.12)
        train_axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        train_axis.set_ylabel(loss_label)

        summaries: List[Tuple[TrainingRun, ConvergenceSummary]] = []
        for run_index, run in enumerate(runs):
            summary = summarize_convergence(run.history, VALIDATION_LOSS_COLUMN, maximize=False)
            summaries.append((run, summary))
            draw_highlight_marker(
                validation_axis, summary.best_epoch, summary.best_value, run.style.color
            )
            annotate_point(
                validation_axis,
                summary.best_epoch,
                summary.best_value,
                f"Min {summary.best_value:.4f} · epoch {summary.best_epoch}",
                (0.0, 46.0 if run_index % 2 == 0 else -40.0),
            )

        add_header(
            figure,
            "Loss convergence",
            "Lowest validation loss  ·  "
            + "  ·  ".join(
                f"{run.style.short_label} {summary.best_value:.4f} at epoch {summary.best_epoch} "
                f"of {summary.total_epochs}"
                for run, summary in summaries
            ),
        )
        add_figure_legend(figure, build_line_handles(runs))
        add_footnote(figure, describe_smoothing(smoothing_window))
        figure.draw_without_rendering()
    return figure


def draw_reference_level(axis: Axes, level: float, label: str) -> None:
    axis.axhline(level, color=TEXT_MUTED_COLOR, linewidth=1.3, linestyle=(0, (5, 4)), zorder=1)
    axis.annotate(
        label,
        xy=(1.0, level),
        xycoords=("axes fraction", "data"),
        xytext=(-4.0, -8.0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=10.5,
        color=TEXT_SECONDARY_COLOR,
    )


def build_dice_subtitle(
    reference: Optional[TrainingRun],
    challengers: Sequence[TrainingRun],
) -> str:
    if reference is None or not challengers:
        return "Mean validation Dice over the three tumor classes"
    reference_summary = summarize_convergence(reference.history)
    sentences = []
    for challenger in challengers:
        challenger_summary = summarize_convergence(challenger.history)
        crossing_epoch = first_epoch_reaching(
            challenger.history, VALIDATION_DICE_COLUMN, reference_summary.best_value
        )
        sentence = (
            f"{challenger.style.short_label} peaks at {challenger_summary.best_value:.4f} "
            f"(epoch {challenger_summary.best_epoch}) vs {reference_summary.best_value:.4f} "
            f"(epoch {reference_summary.best_epoch}) for {reference.style.short_label}"
        )
        if crossing_epoch is not None:
            sentence += (
                f"  ·  it overtakes the {reference.style.short_label.lower()} peak "
                f"at epoch {crossing_epoch}"
            )
        sentences.append(sentence)
    return "  |  ".join(sentences)


def plot_validation_dice(
    runs: Sequence[TrainingRun],
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    reference_name: str = REFERENCE_MODEL,
) -> Figure:
    reference = find_run(runs, reference_name)
    challengers = [run for run in runs if run.name != reference_name]
    with neon_theme():
        figure, (axis,) = create_figure((14.0, 7.4), columns=1)
        for run in runs:
            plot_metric_trace(axis, run, VALIDATION_DICE_COLUMN, smoothing_window)
        style_epoch_axis(axis, runs, None)

        summaries = {run.name: summarize_convergence(run.history) for run in runs}
        highest_dice = max(summary.best_value for summary in summaries.values())
        axis.set_ylim(0.0, min(1.0, highest_dice + 0.16))
        axis.yaxis.set_major_locator(MultipleLocator(0.1))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.set_ylabel("Validation Dice (mean of 3 tumor classes)")

        if reference is not None:
            reference_summary = summaries[reference.name]
            draw_reference_level(
                axis,
                reference_summary.best_value,
                f"{reference.style.short_label} best  {reference_summary.best_value:.4f}",
            )
            for challenger in challengers:
                crossing_epoch = first_epoch_reaching(
                    challenger.history, VALIDATION_DICE_COLUMN, reference_summary.best_value
                )
                if crossing_epoch is None:
                    continue
                crossing_value = float(
                    challenger.history.loc[
                        challenger.history[EPOCH_COLUMN] == crossing_epoch, VALIDATION_DICE_COLUMN
                    ].iloc[0]
                )
                draw_point_marker(axis, crossing_epoch, crossing_value, challenger.style.color)
                crossing_seconds = seconds_until_epoch(challenger.history, crossing_epoch)
                time_suffix = (
                    ""
                    if crossing_seconds is None
                    else f" · {crossing_seconds / SECONDS_PER_MINUTE:.0f} min"
                )
                annotate_point(
                    axis,
                    crossing_epoch,
                    crossing_value,
                    f"Passes {reference.style.short_label.lower()} best at epoch "
                    f"{crossing_epoch}{time_suffix}",
                    (34.0, 52.0),
                    horizontal_alignment="left",
                )

        for run in runs:
            summary = summaries[run.name]
            draw_highlight_marker(axis, summary.best_epoch, summary.best_value, run.style.color)
            is_reference = run.name == reference_name
            annotate_point(
                axis,
                summary.best_epoch,
                summary.best_value,
                format_epoch_label(
                    "Best", summary.best_value, summary.best_epoch, summary.seconds_to_best
                ),
                (60.0, -62.0) if is_reference else (0.0, 46.0),
                horizontal_alignment="left" if is_reference else "center",
            )

        add_header(figure, "Validation Dice per epoch", build_dice_subtitle(reference, challengers))
        add_figure_legend(figure, build_line_handles(runs))
        add_footnote(figure, describe_smoothing(smoothing_window))
        figure.draw_without_rendering()
    return figure


def ordered_models(test_metrics: pd.DataFrame) -> List[str]:
    present_models = list(dict.fromkeys(test_metrics["model"]))
    for name in present_models:
        resolve_model_style(name)
    return [name for name in MODEL_ORDER if name in present_models]


def ordered_regions(test_metrics: pd.DataFrame) -> List[str]:
    present_regions = list(dict.fromkeys(test_metrics["region"]))
    known_regions = [region for region in REGION_LABELS if region in present_regions]
    return known_regions + [region for region in present_regions if region not in REGION_LABELS]


def compute_group_positions(regions: Sequence[str]) -> np.ndarray:
    aggregate_shift = np.cumsum(
        [
            (
                AGGREGATE_GROUP_OFFSET
                if region in AGGREGATE_REGIONS
                and (index == 0 or regions[index - 1] not in AGGREGATE_REGIONS)
                else 0.0
            )
            for index, region in enumerate(regions)
        ]
    )
    return np.arange(len(regions), dtype=float) + aggregate_shift


def lookup_score(test_metrics: pd.DataFrame, model: str, region: str, metric: str) -> float:
    matches = test_metrics.loc[
        (test_metrics["model"] == model) & (test_metrics["region"] == region), metric
    ]
    return float(matches.iloc[0]) if not matches.empty else float("nan")


def format_region_label(region: str) -> str:
    return REGION_LABELS.get(region, region.replace("_", " ").capitalize())


def draw_grouped_columns(
    axis: Axes,
    test_metrics: pd.DataFrame,
    models: Sequence[str],
    regions: Sequence[str],
    metric: str,
) -> None:
    positions = compute_group_positions(regions)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * (BAR_WIDTH + BAR_GAP)
    scores = {
        model: np.array([lookup_score(test_metrics, model, region, metric) for region in regions])
        for model in models
    }
    for model, offset in zip(models, offsets):
        color = resolve_model_style(model).color
        axis.bar(
            positions + offset,
            np.nan_to_num(scores[model]),
            width=BAR_WIDTH,
            color=color,
            linewidth=0.0,
            zorder=3,
        )
        for position, score in zip(positions + offset, scores[model]):
            if np.isfinite(score):
                axis.text(
                    position,
                    score + 0.018,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                    color=TEXT_PRIMARY_COLOR,
                    zorder=4,
                )

    if len(models) == 2:
        reference_scores, challenger_scores = (scores[model] for model in models)
        for position, reference_score, challenger_score in zip(
            positions, reference_scores, challenger_scores
        ):
            if not (np.isfinite(reference_score) and np.isfinite(challenger_score)):
                continue
            axis.text(
                position,
                max(reference_score, challenger_score) + 0.105,
                f"{challenger_score - reference_score:+.3f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="semibold",
                color=TEXT_PRIMARY_COLOR,
                bbox={
                    "boxstyle": "round,pad=0.35,rounding_size=0.6",
                    "facecolor": RAISED_SURFACE_COLOR,
                    "edgecolor": AXIS_COLOR,
                    "linewidth": 0.9,
                },
                zorder=5,
            )

    first_aggregate = next(
        (index for index, region in enumerate(regions) if region in AGGREGATE_REGIONS), None
    )
    if first_aggregate is not None and first_aggregate > 0:
        axis.axvline(
            (positions[first_aggregate - 1] + positions[first_aggregate]) / 2.0,
            color=AXIS_COLOR,
            linewidth=1.1,
            zorder=1,
        )

    axis.set_xticks(positions, [format_region_label(region) for region in regions])
    axis.set_xlim(positions[0] - 0.65, positions[-1] + 0.65)
    axis.set_ylim(0.0, 1.08)
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="both", length=0)


def build_test_subtitle(test_metrics: pd.DataFrame, models: Sequence[str]) -> str:
    if len(models) != 2:
        return "Dice and IoU per tumor class on the held-out test set"
    reference, challenger = models
    fragments = []
    for region, description in (("mean", "mean Dice"), ("whole_tumor", "whole-tumor Dice")):
        reference_score = lookup_score(test_metrics, reference, region, "dice")
        challenger_score = lookup_score(test_metrics, challenger, region, "dice")
        if np.isfinite(reference_score) and np.isfinite(challenger_score):
            fragments.append(
                f"{description} {reference_score:.3f} → {challenger_score:.3f} "
                f"({challenger_score - reference_score:+.3f})"
            )
    prefix = (
        f"{resolve_model_style(challenger).short_label} vs "
        f"{resolve_model_style(reference).short_label} on unseen patients"
    )
    return "  ·  ".join([prefix, *fragments])


def plot_test_metrics(test_metrics: pd.DataFrame) -> Figure:
    models = ordered_models(test_metrics)
    regions = ordered_regions(test_metrics)
    with neon_theme():
        figure, axes = create_figure((15.0, 6.9), columns=2, share_y=True)
        for axis, metric, title in zip(axes, ("dice", "iou"), ("Dice", "IoU")):
            draw_grouped_columns(axis, test_metrics, models, regions, metric)
            axis.set_title(title)
        axes[0].set_ylabel("Score on the test set")
        add_header(
            figure, "Held-out test set performance", build_test_subtitle(test_metrics, models)
        )
        add_figure_legend(figure, build_patch_handles(models))
        add_footnote(
            figure,
            "Scores accumulated over every test pixel per class (background excluded)  ·  "
            "mean = macro average of the three tumor classes\n"
            "Whole tumor = all tumor classes merged into one foreground label  ·  chips show "
            f"{' − '.join(resolve_model_style(model).short_label for model in reversed(models))}",
        )
        figure.draw_without_rendering()
    return figure


def save_figure(
    figure: Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str] = ("png",),
    dpi: int = DEFAULT_DPI,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    with neon_theme():
        for file_format in formats:
            path = output_dir / f"{stem}.{file_format.lstrip('.')}"
            figure.savefig(
                path,
                dpi=dpi,
                facecolor=figure.get_facecolor(),
                bbox_inches="tight",
                pad_inches=0.2,
            )
            saved_paths.append(path)
    return saved_paths
