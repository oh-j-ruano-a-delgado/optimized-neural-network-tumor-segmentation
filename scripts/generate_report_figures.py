"""Generate the vector PDF figures of the LaTeX report in ``docs/report/figures``.

Figures (``--figures`` selects a subset; every file is overwritten on each run):
    dataset_split            ``dataset_split.pdf``. Left: slices per tumor class as horizontal
                             bars stacked by subset (training, validation, test), with the slice
                             and patient totals of each class. Right: mean share of the slice
                             area that the tumor occupies per class, the pixel-level imbalance
                             that motivates the loss design.
    training_curves          ``training_curves.pdf``. One column per run (A and E) and three rows:
                             training and validation loss, mean validation Dice, and the learning
                             rate applied at each epoch.
    confusion_matrices       ``conf_matrix_E.pdf`` and ``conf_matrix_F.pdf``. Row-normalized pixel
                             confusion matrices of Runs E and F on the test set, drawn with
                             ``seaborn.heatmap``.
    mri_examples             ``mri_examples.pdf``. One raw slice per tumor type (top) and a
                             magnified view of the lesion with the expert contour (bottom).
    segmentation_comparison  ``segmentation_comparison.pdf``. One glioma test slice as a 1 x 4
                             grid: input MRI, expert mask, Run A prediction, Run E prediction.

Theme:
    The figures reuse the theme of ``src.utils.visualization`` (neutral ink, embedded TrueType
    fonts) with four changes for a printed A4 page. The page surface is white, so that Cloud
    Dancer can act as a visible shading instead of being the background. Font sizes are set for
    figures drawn at their final width (6.3 inches, the text width of the report), so the report
    includes them at ``\\linewidth`` without rescaling the type. Grid lines are solid hairlines
    instead of dashes. The typeface is STIX General, a Times-like serif that ships with
    matplotlib, instead of the first installed family of the shared theme: every machine
    (Windows, Kaggle, Linux) then draws the same glyphs, so regenerating a single figure never
    leaves it in a different typeface from the others.
    Palette (Pantone, exact HEX codes): Classic Blue #0F4C81 marks training curves, the training
    subset and the expert mask; Viva Magenta #BB2649 marks validation curves, the validation
    subset, model predictions and the highlighted confusion cell; Cloud Dancer #F0EEE9 fills the
    test subset, the band after the best epoch, the low end of the heat map ramp, pixels predicted
    as another tumor type and the pending panels. The band is drawn fully opaque: on a white page
    a translucent Cloud Dancer band would not be visible in print. The test subset is the only
    light bar segment, so it gets a thin neutral outline. Text always uses neutral ink, never a
    series color.
    ``CreationDate`` is removed from the PDF metadata so that regenerating a figure from unchanged
    inputs does not produce a different file.

Dataset split:
    The three subsets are rebuilt with the functions that training uses
    (``discover_samples`` and ``split_samples_by_patient`` with the default validation fraction
    0.2 and seed 42), so the chart cannot drift from the real partition. The tumor share of a
    slice is the fraction of non-zero pixels of its raw mask, averaged per class over all 3,064
    slices.

Training curves:
    The best epoch is the epoch with the highest mean validation Dice, which is the epoch whose
    checkpoint training keeps. A dotted vertical line marks it and a Cloud Dancer band spans from
    it to the last epoch. Neither Run A nor Run E used early stopping: both trained their whole
    epoch budget (100 and 150 epochs) and kept the best checkpoint. The band therefore shows the
    epochs that brought no new best, that is, the patience window an early-stopping rule would
    have needed to cover, and the figure labels it as epochs without improvement instead of
    claiming that training stopped early.
    Run A trains with the Dice-Focal loss and Run E with the Focal Tversky-Focal loss. Their
    values are not on a common scale, so each run has its own loss axis and only the Dice row
    shares its axis between the two runs. The two columns are as wide as their epoch counts, so
    one epoch has the same width in both.
    Histories written before the cosine schedule existed (Run A) have no ``learning_rate``
    column. Those runs used the constant rate found by the genetic algorithm, which is read from
    ``optimization/best_hyperparameters.json`` in the runs directory. The Dice row draws the raw
    per-epoch values as a thin translucent trace under a 5-epoch centered rolling mean; the
    marker and its label use the raw value of the best epoch.

Confusion matrices:
    ``transfer_learning/15_test_audit_E_vs_F/slice_confusions.npz`` stores, for Runs E and F, one
    4 x 4 pixel confusion matrix per test slice (rows are the expert label, columns the
    prediction, in the order background, meningioma, glioma, pituitary), in the order in which
    ``discover_samples`` lists the Testing split. Summing over slices gives the test-set matrix,
    and each row is divided by its total. The script checks that the tumor row of every slice
    matches the class of the sample at the same position and stops if the file and the dataset
    disagree. A Viva Magenta frame marks the dominant error of both models: glioma pixels
    predicted as background.

MRI examples:
    The three slices are fixed by name (``MRI_EXAMPLES``) so the figure is reproducible. They are
    axial Training slices chosen by eye to show the delineation problem: a meningioma and a
    pituitary tumor with sharp borders, and a glioma whose expert contour encloses tissue with
    almost no visible edge. Intensities are only rescaled for display (percentile clipping and
    min-max, as in the evaluation transform); the slices are not cropped or resized.

Segmentation comparison:
    The slice is the glioma test slice whose Run E Dice is closest to the median Run E Dice over
    the glioma test slices (computed from the audit file above), so it is a typical case for the
    final model and not a hand-picked success. It is preprocessed with the same deterministic
    steps as ``SegmentationTransform`` without augmentation (grayscale, margin crop, resize,
    min-max normalization), and each checkpoint predicts it with a single forward pass on the
    CPU. Prediction panels fill the predicted glioma pixels in Viva Magenta, fill pixels predicted
    as another tumor type in Cloud Dancer, and draw the expert contour in Classic Blue. Their
    titles report the glioma Dice of that single slice.
    No run directory stores predicted masks (the diagnostics folders only hold tables), so the
    predictions always come from the checkpoints. When a checkpoint file is missing, or
    ``--skip-inference`` is passed, the corresponding panel is drawn as a Cloud Dancer placeholder
    that says the prediction is pending, and the script prints which panels are placeholders.

Usage (from the project root):
    python -m scripts.generate_report_figures [--data-dir data/raw] [--runs-dir runs]
        [--output-dir docs/report/figures] [--validation-fraction 0.2] [--seed 42]
        [--figures dataset_split training_curves confusion_matrices mri_examples
                   segmentation_comparison]
        [--skip-inference]
"""

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Final, Iterator, List, Mapping, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from numpy.typing import NDArray

from src.dataset.constants import (
    BACKGROUND_LABEL,
    IMAGES_DIRNAME,
    MASKS_DIRNAME,
    RAW_DATA_DIR,
    TESTING_SPLIT,
    TRAINING_SPLIT,
    TUMOR_CLASS_NAMES,
    parse_patient_id,
)
from src.dataset.dataloader import (
    SegmentationSample,
    discover_samples,
    read_image_file,
    split_samples_by_patient,
)
from src.dataset.transforms import (
    DEFAULT_IMAGE_SIZE,
    crop_background_margins,
    normalize_intensity,
    rescale_min_max,
    resize_image_and_mask,
    to_grayscale,
)
from src.utils.checkpoint import build_model_from_checkpoint, load_training_checkpoint
from src.utils.visualization import (
    AXIS_COLOR,
    CLASSIC_BLUE,
    CLOUD_DANCER,
    DEFAULT_SMOOTHING_WINDOW,
    EPOCH_COLUMN,
    TEXT_MUTED_COLOR,
    TEXT_PRIMARY_COLOR,
    TEXT_SECONDARY_COLOR,
    TRAIN_LOSS_COLUMN,
    VALIDATION_DICE_COLUMN,
    VALIDATION_LOSS_COLUMN,
    VIVA_MAGENTA,
    build_theme,
    load_history,
    rolling_mean,
    summarize_convergence,
)

DEFAULT_RUNS_DIR: Final[Path] = Path("runs")
DEFAULT_OUTPUT_DIR: Final[Path] = Path("docs/report/figures")
DEFAULT_VALIDATION_FRACTION: Final[float] = 0.2
DEFAULT_SEED: Final[int] = 42
HISTORY_FILENAME: Final[str] = "history.csv"
CHECKPOINT_FILENAME: Final[str] = "best_model.pth"
SEARCH_RESULT_PATH: Final[Path] = Path("optimization/best_hyperparameters.json")
AUDIT_CONFUSIONS_PATH: Final[Path] = Path(
    "transfer_learning/15_test_audit_E_vs_F/slice_confusions.npz"
)
LEARNING_RATE_COLUMN: Final[str] = "learning_rate"

PAGE_COLOR: Final[str] = "#FFFFFF"
ON_COLOR_TEXT: Final[str] = "#FFFFFF"
TEXT_WIDTH_INCHES: Final[float] = 6.3
HALF_WIDTH_INCHES: Final[float] = 3.05
RASTER_DPI: Final[int] = 300
SMALL_FONT_SIZE: Final[float] = 7.5
LEARNING_RATE_SCALE: Final[float] = 1e4
BAR_HEIGHT: Final[float] = 0.62
SHARE_BAR_HEIGHT: Final[float] = 0.3
SEGMENT_GAP_WIDTH: Final[float] = 1.0
MASK_ALPHA: Final[float] = 0.6
OTHER_TUMOR_ALPHA: Final[float] = 0.9
CONTOUR_WIDTH: Final[float] = 1.1
ZOOM_MARGIN_FACTOR: Final[float] = 1.7
MINIMUM_ZOOM_HALF_SIDE: Final[int] = 44

BACKGROUND_DISPLAY_NAME: Final[str] = "Fondo"
CLASS_DISPLAY_NAMES: Final[Mapping[int, str]] = {
    1: "Meningioma",
    2: "Glioma",
    3: "Hipofisario",
}
GLIOMA_LABEL: Final[int] = 2
CONFUSION_LABELS: Final[Tuple[str, ...]] = (
    BACKGROUND_DISPLAY_NAME,
    *(CLASS_DISPLAY_NAMES[label] for label in sorted(CLASS_DISPLAY_NAMES)),
)

FIGURE_NAMES: Final[Tuple[str, ...]] = (
    "dataset_split",
    "training_curves",
    "confusion_matrices",
    "mri_examples",
    "segmentation_comparison",
)

REPORT_THEME_OVERRIDES: Final[Mapping[str, Any]] = {
    "figure.facecolor": PAGE_COLOR,
    "figure.edgecolor": PAGE_COLOR,
    "savefig.facecolor": PAGE_COLOR,
    "savefig.edgecolor": PAGE_COLOR,
    "axes.facecolor": PAGE_COLOR,
    "axes.linewidth": 0.6,
    "axes.labelsize": 8.5,
    "axes.labelpad": 4.0,
    "axes.titlesize": 9.0,
    "axes.titlepad": 6.0,
    "font.family": ["STIXGeneral", "DejaVu Serif"],
    "font.size": 8.5,
    "mathtext.fontset": "stix",
    "xtick.labelsize": SMALL_FONT_SIZE,
    "ytick.labelsize": SMALL_FONT_SIZE,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.pad": 3.0,
    "ytick.major.pad": 3.0,
    "legend.fontsize": SMALL_FONT_SIZE,
    "legend.handlelength": 1.8,
    "legend.columnspacing": 1.4,
    "grid.linestyle": "-",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.55,
    "lines.linewidth": 1.4,
}


@dataclass(frozen=True)
class DatasetSubset:
    label: str
    color: str
    samples: Tuple[SegmentationSample, ...]

    def count_slices(self, class_label: int) -> int:
        return sum(sample.class_label == class_label for sample in self.samples)


@dataclass(frozen=True)
class TrainingRunSpec:
    name: str
    directory: Path
    description: str
    loss_label: str

    def history_path(self, runs_dir: Path) -> Path:
        return runs_dir / self.directory / HISTORY_FILENAME

    def checkpoint_path(self, runs_dir: Path) -> Path:
        return runs_dir / self.directory / CHECKPOINT_FILENAME


@dataclass(frozen=True)
class ConfusionMatrixSpec:
    run_name: str
    description: str

    @property
    def stem(self) -> str:
        return f"conf_matrix_{self.run_name}"


@dataclass(frozen=True)
class MriExample:
    class_label: int
    stem: str
    title: str
    descriptor: str


@dataclass(frozen=True)
class RunPrediction:
    run_name: str
    class_mask: Optional[NDArray[np.uint8]]


@dataclass(frozen=True)
class ZoomWindow:
    top: int
    bottom: int
    left: int
    right: int

    def crop(self, array: NDArray[np.generic]) -> NDArray[np.generic]:
        return array[self.top : self.bottom, self.left : self.right]


RUN_A: Final[TrainingRunSpec] = TrainingRunSpec(
    name="A",
    directory=Path("baseline_vs_ga/02_A_ga_optimized"),
    description=r"$\eta$ constante, $F=16$",
    loss_label="Pérdida Dice-Focal",
)
RUN_E: Final[TrainingRunSpec] = TrainingRunSpec(
    name="E",
    directory=Path("capacity_and_lr_schedule/10_E_ultimate_unet"),
    description=r"$\mathit{cosine\ annealing}$, $F=32$",
    loss_label="Pérdida Focal Tversky-Focal",
)
TRAINING_RUNS: Final[Tuple[TrainingRunSpec, ...]] = (RUN_A, RUN_E)
CONFUSION_MATRICES: Final[Tuple[ConfusionMatrixSpec, ...]] = (
    ConfusionMatrixSpec("E", "U-Net desde cero"),
    ConfusionMatrixSpec("F", "ResNet34 preentrenado"),
)
MRI_EXAMPLES: Final[Tuple[MriExample, ...]] = (
    MriExample(1, "108945_116", "Meningioma", "bordes nítidos, realce intenso"),
    MriExample(2, "MR049019G_2134", "Glioma", "márgenes difusos e infiltrativos"),
    MriExample(3, "103731_1335", "Tumor hipofisario", "lesión pequeña, bordes nítidos"),
)


@contextmanager
def report_theme() -> Iterator[None]:
    with matplotlib.rc_context({**build_theme(), **REPORT_THEME_OVERRIDES}):
        yield


def save_pdf(figure: Figure, output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.pdf"
    with report_theme():
        figure.savefig(
            path,
            format="pdf",
            dpi=RASTER_DPI,
            bbox_inches="tight",
            pad_inches=0.03,
            metadata={"CreationDate": None},
        )
    return path


def style_image_axis(axis: Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def show_slice(axis: Axes, image: NDArray[np.float32]) -> None:
    axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bicubic")
    style_image_axis(axis)


def fill_region(
    axis: Axes, region: NDArray[np.bool_], color: str, alpha: float = MASK_ALPHA
) -> None:
    if not region.any():
        return
    axis.imshow(
        np.ma.masked_where(~region, region.astype(np.float32)),
        cmap=ListedColormap([color]),
        vmin=0.0,
        vmax=1.0,
        alpha=alpha,
        interpolation="nearest",
    )


def outline_region(axis: Axes, region: NDArray[np.bool_], color: str) -> None:
    if not region.any():
        return
    axis.contour(
        region.astype(np.float32), levels=[0.5], colors=[color], linewidths=CONTOUR_WIDTH
    )


def load_dataset_subsets(
    data_dir: Path, validation_fraction: float, seed: int
) -> Tuple[DatasetSubset, ...]:
    training_samples, validation_samples = split_samples_by_patient(
        discover_samples(data_dir / TRAINING_SPLIT), validation_fraction, seed
    )
    testing_samples = discover_samples(data_dir / TESTING_SPLIT)
    return (
        DatasetSubset("Entrenamiento", CLASSIC_BLUE, tuple(training_samples)),
        DatasetSubset("Validación", VIVA_MAGENTA, tuple(validation_samples)),
        DatasetSubset("Prueba", CLOUD_DANCER, tuple(testing_samples)),
    )


def count_patients(subsets: Sequence[DatasetSubset], class_label: int) -> int:
    return len(
        {
            sample.patient_id
            for subset in subsets
            for sample in subset.samples
            if sample.class_label == class_label
        }
    )


def measure_tumor_share(sample: SegmentationSample) -> float:
    mask = read_image_file(sample.mask_path, cv2.IMREAD_GRAYSCALE)
    return float(np.count_nonzero(mask != BACKGROUND_LABEL)) / float(mask.size)


def measure_tumor_shares(subsets: Sequence[DatasetSubset]) -> Dict[int, List[float]]:
    shares: Dict[int, List[float]] = {label: [] for label in sorted(TUMOR_CLASS_NAMES)}
    for subset in subsets:
        for sample in subset.samples:
            shares[sample.class_label].append(measure_tumor_share(sample))
    return shares


def draw_split_bars(axis: Axes, subsets: Sequence[DatasetSubset]) -> None:
    class_labels = sorted(TUMOR_CLASS_NAMES)
    positions = np.arange(len(class_labels), dtype=float)
    segment_starts = np.zeros(len(class_labels))
    for subset in subsets:
        counts = np.array([subset.count_slices(label) for label in class_labels], dtype=float)
        is_light = subset.color == CLOUD_DANCER
        axis.barh(
            positions,
            counts,
            left=segment_starts,
            height=BAR_HEIGHT,
            color=subset.color,
            edgecolor=AXIS_COLOR if is_light else PAGE_COLOR,
            linewidth=0.6 if is_light else SEGMENT_GAP_WIDTH,
            zorder=3,
        )
        for position, start, count in zip(positions, segment_starts, counts):
            axis.text(
                start + count / 2.0,
                position,
                f"{int(count)}",
                ha="center",
                va="center",
                fontsize=SMALL_FONT_SIZE,
                color=TEXT_PRIMARY_COLOR if is_light else ON_COLOR_TEXT,
                zorder=4,
            )
        segment_starts = segment_starts + counts

    for position, label, total in zip(positions, class_labels, segment_starts):
        axis.text(
            total + 0.012 * segment_starts.max(),
            position,
            f"{int(total)} cortes\n{count_patients(subsets, label)} pacientes",
            ha="left",
            va="center",
            fontsize=SMALL_FONT_SIZE,
            color=TEXT_SECONDARY_COLOR,
            linespacing=1.25,
        )

    axis.set_yticks(positions, [CLASS_DISPLAY_NAMES[label] for label in class_labels])
    axis.set_ylim(len(class_labels) - 0.5, -0.5)
    axis.set_xlim(0.0, segment_starts.max() * 1.27)
    axis.xaxis.set_major_locator(MultipleLocator(250))
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", visible=True)
    axis.tick_params(axis="y", length=0)
    axis.set_xlabel("Cortes de RM")
    axis.set_title("Cortes por clase y subconjunto")


def draw_tumor_share_bars(axis: Axes, shares: Mapping[int, Sequence[float]]) -> None:
    class_labels = sorted(shares)
    positions = np.arange(len(class_labels), dtype=float)
    mean_percentages = np.array([100.0 * float(np.mean(shares[label])) for label in class_labels])
    overall_percentage = 100.0 * float(
        np.mean([share for label in class_labels for share in shares[label]])
    )
    axis.barh(
        positions,
        mean_percentages,
        height=SHARE_BAR_HEIGHT,
        color=CLASSIC_BLUE,
        linewidth=0.0,
        zorder=3,
    )
    for position, percentage in zip(positions, mean_percentages):
        axis.text(
            percentage + 0.08,
            position,
            f"{percentage:.1f} %",
            ha="left",
            va="center",
            fontsize=SMALL_FONT_SIZE,
            color=TEXT_PRIMARY_COLOR,
        )
    axis.axvline(overall_percentage, color=TEXT_MUTED_COLOR, linewidth=0.8, linestyle=":", zorder=2)
    axis.text(
        overall_percentage + 0.07,
        (positions[-2] + positions[-1]) / 2.0,
        f"media global\n{overall_percentage:.1f} %",
        ha="left",
        va="center",
        fontsize=SMALL_FONT_SIZE,
        color=TEXT_SECONDARY_COLOR,
        linespacing=1.25,
    )
    axis.set_xlim(0.0, mean_percentages.max() * 1.38)
    axis.xaxis.set_major_locator(MultipleLocator(1.0))
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", visible=True)
    axis.tick_params(axis="y", length=0, labelleft=False)
    axis.set_xlabel("Área tumoral media (% del corte)")
    axis.set_title("Desequilibrio a nivel de píxel")


def plot_dataset_split(
    subsets: Sequence[DatasetSubset], shares: Mapping[int, Sequence[float]]
) -> Figure:
    with report_theme():
        figure = Figure(figsize=(TEXT_WIDTH_INCHES, 2.55), layout="constrained")
        count_axis, share_axis = figure.subplots(1, 2, sharey=True, width_ratios=[2.35, 1.0])
        draw_split_bars(count_axis, subsets)
        draw_tumor_share_bars(share_axis, shares)
        figure.legend(
            handles=[
                Patch(
                    facecolor=subset.color,
                    edgecolor=AXIS_COLOR if subset.color == CLOUD_DANCER else subset.color,
                    linewidth=0.6,
                    label=f"{subset.label} ({len(subset.samples)} cortes)",
                )
                for subset in subsets
            ],
            loc="outside upper left",
            ncol=len(subsets),
        )
        figure.draw_without_rendering()
    return figure


def load_searched_learning_rate(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Genetic search result '{path}' does not exist.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["best_hyperparameters"][LEARNING_RATE_COLUMN])


def resolve_learning_rates(history: pd.DataFrame, constant_learning_rate: float) -> pd.Series:
    if LEARNING_RATE_COLUMN in history:
        return history[LEARNING_RATE_COLUMN].astype(float)
    return pd.Series(constant_learning_rate, index=history.index, dtype=float)


def mark_best_epoch(axis: Axes, best_epoch: int, last_epoch: int) -> None:
    axis.axvspan(best_epoch, last_epoch, facecolor=CLOUD_DANCER, linewidth=0.0, zorder=0)
    axis.axvline(best_epoch, color=TEXT_PRIMARY_COLOR, linewidth=0.9, linestyle=":", zorder=4)


def draw_loss_panel(axis: Axes, history: pd.DataFrame, best_epoch: int, last_epoch: int) -> None:
    epochs = history[EPOCH_COLUMN]
    mark_best_epoch(axis, best_epoch, last_epoch)
    axis.plot(epochs, history[TRAIN_LOSS_COLUMN], color=CLASSIC_BLUE, zorder=3)
    axis.plot(epochs, history[VALIDATION_LOSS_COLUMN], color=VIVA_MAGENTA, zorder=3)
    highest_loss = float(history[[TRAIN_LOSS_COLUMN, VALIDATION_LOSS_COLUMN]].max().max())
    axis.set_ylim(0.0, highest_loss * 1.06)
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.text(
        (best_epoch + last_epoch) / 2.0,
        0.93,
        f"{last_epoch - best_epoch} épocas\nsin mejora",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=SMALL_FONT_SIZE,
        color=TEXT_SECONDARY_COLOR,
        linespacing=1.25,
    )


def draw_dice_panel(
    axis: Axes, history: pd.DataFrame, best_epoch: int, best_value: float, last_epoch: int
) -> None:
    epochs = history[EPOCH_COLUMN]
    mark_best_epoch(axis, best_epoch, last_epoch)
    axis.plot(
        epochs,
        history[VALIDATION_DICE_COLUMN],
        color=VIVA_MAGENTA,
        linewidth=0.7,
        alpha=0.4,
        zorder=2,
    )
    axis.plot(
        epochs,
        rolling_mean(history[VALIDATION_DICE_COLUMN], DEFAULT_SMOOTHING_WINDOW),
        color=VIVA_MAGENTA,
        zorder=3,
    )
    axis.plot(
        [best_epoch],
        [best_value],
        linestyle="none",
        marker="o",
        markersize=5.5,
        markerfacecolor=VIVA_MAGENTA,
        markeredgecolor=PAGE_COLOR,
        markeredgewidth=1.0,
        zorder=6,
    )
    axis.annotate(
        f"{best_value:.4f} (época {best_epoch})",
        xy=(best_epoch, best_value),
        xytext=(-5.0, 7.0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=SMALL_FONT_SIZE,
        color=TEXT_PRIMARY_COLOR,
        zorder=7,
    )
    axis.set_ylim(0.2, 0.9)
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def draw_learning_rate_panel(
    axis: Axes, history: pd.DataFrame, learning_rates: pd.Series, best_epoch: int, last_epoch: int
) -> None:
    mark_best_epoch(axis, best_epoch, last_epoch)
    axis.plot(
        history[EPOCH_COLUMN],
        learning_rates * LEARNING_RATE_SCALE,
        color=TEXT_SECONDARY_COLOR,
        linewidth=1.2,
        zorder=3,
    )
    axis.set_ylim(0.0, float(learning_rates.max()) * LEARNING_RATE_SCALE * 1.2)
    axis.yaxis.set_major_locator(MultipleLocator(1.5))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.set_xlim(0, last_epoch)
    axis.xaxis.set_major_locator(MultipleLocator(25))
    axis.set_xlabel("Época")


def build_training_legend_handles() -> List[Any]:
    return [
        Line2D([], [], color=CLASSIC_BLUE, label="Entrenamiento"),
        Line2D([], [], color=VIVA_MAGENTA, label="Validación"),
        Line2D(
            [],
            [],
            color=TEXT_PRIMARY_COLOR,
            linewidth=0.9,
            linestyle=":",
            label="Mejor punto de control",
        ),
        Patch(
            facecolor=CLOUD_DANCER,
            edgecolor=AXIS_COLOR,
            linewidth=0.5,
            label="Épocas posteriores sin mejora",
        ),
    ]


def plot_training_curves(
    runs: Sequence[TrainingRunSpec],
    histories: Sequence[pd.DataFrame],
    constant_learning_rate: float,
) -> Figure:
    last_epochs = [int(history[EPOCH_COLUMN].max()) for history in histories]
    with report_theme():
        figure = Figure(figsize=(TEXT_WIDTH_INCHES, 4.9), layout="constrained")
        axes = figure.subplots(
            3,
            len(runs),
            sharex="col",
            squeeze=False,
            height_ratios=[3.0, 3.0, 1.15],
            width_ratios=last_epochs,
        )
        for column, (run, history, last_epoch) in enumerate(zip(runs, histories, last_epochs)):
            loss_axis, dice_axis, rate_axis = axes[:, column]
            summary = summarize_convergence(history)
            draw_loss_panel(loss_axis, history, summary.best_epoch, last_epoch)
            draw_dice_panel(
                dice_axis, history, summary.best_epoch, summary.best_value, last_epoch
            )
            draw_learning_rate_panel(
                rate_axis,
                history,
                resolve_learning_rates(history, constant_learning_rate),
                summary.best_epoch,
                last_epoch,
            )
            loss_axis.set_title(f"Ejecución {run.name}")
            loss_axis.text(
                1.0,
                1.035,
                run.description,
                transform=loss_axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=SMALL_FONT_SIZE,
                color=TEXT_SECONDARY_COLOR,
            )
            loss_axis.set_ylabel(run.loss_label)
            if column > 0:
                dice_axis.sharey(axes[1, 0])
                rate_axis.sharey(axes[2, 0])
                dice_axis.tick_params(labelleft=False)
                rate_axis.tick_params(labelleft=False)
        axes[1, 0].set_ylabel("Dice medio de validación")
        axes[2, 0].set_ylabel(r"$\eta$ ($\times10^{-4}$)")
        figure.legend(
            handles=build_training_legend_handles(), loc="outside upper center", ncol=4
        )
        figure.draw_without_rendering()
    return figure


def load_audit_confusions(path: Path, run_name: str) -> NDArray[np.int64]:
    if not path.is_file():
        raise FileNotFoundError(f"Test audit file '{path}' does not exist.")
    with np.load(path) as archive:
        if run_name not in archive.files:
            raise ValueError(f"Test audit file '{path}' has no matrices for run '{run_name}'.")
        return np.asarray(archive[run_name], dtype=np.int64)


def validate_audit_alignment(
    confusions: NDArray[np.int64], samples: Sequence[SegmentationSample]
) -> None:
    if confusions.shape[0] != len(samples):
        raise ValueError(
            f"The test audit holds {confusions.shape[0]} slices but the test set has "
            f"{len(samples)}."
        )
    audited_labels = confusions[:, 1:, :].sum(axis=2).argmax(axis=1) + 1
    sample_labels = np.array([sample.class_label for sample in samples])
    if not np.array_equal(audited_labels, sample_labels):
        raise ValueError("The test audit and the test set list their slices in different orders.")


def normalize_rows(counts: NDArray[np.int64]) -> NDArray[np.float64]:
    row_totals = counts.sum(axis=1, keepdims=True).astype(np.float64)
    return np.divide(
        counts.astype(np.float64),
        row_totals,
        out=np.zeros(counts.shape, dtype=np.float64),
        where=row_totals > 0,
    )


def plot_confusion_matrix(row_shares: NDArray[np.float64], spec: ConfusionMatrixSpec) -> Figure:
    percentages = 100.0 * row_shares
    annotations = np.array([[f"{value:.1f}" for value in row] for row in percentages])
    with report_theme():
        figure = Figure(figsize=(HALF_WIDTH_INCHES, 3.0), layout="constrained")
        axis = figure.subplots()
        sns.heatmap(
            percentages,
            ax=axis,
            annot=annotations,
            fmt="",
            cmap=LinearSegmentedColormap.from_list("cloud_to_blue", [CLOUD_DANCER, CLASSIC_BLUE]),
            vmin=0.0,
            vmax=100.0,
            cbar=False,
            square=True,
            linewidths=1.5,
            linecolor=PAGE_COLOR,
            xticklabels=CONFUSION_LABELS,
            yticklabels=CONFUSION_LABELS,
            annot_kws={"fontsize": 8.5},
        )
        glioma_row = CONFUSION_LABELS.index(CLASS_DISPLAY_NAMES[GLIOMA_LABEL])
        background_column = CONFUSION_LABELS.index(BACKGROUND_DISPLAY_NAME)
        axis.add_patch(
            Rectangle(
                (background_column + 0.04, glioma_row + 0.04),
                0.92,
                0.92,
                fill=False,
                edgecolor=VIVA_MAGENTA,
                linewidth=1.8,
                zorder=5,
            )
        )
        axis.grid(False)
        axis.tick_params(axis="both", length=0)
        axis.set_xticklabels(axis.get_xticklabels(), rotation=0, fontsize=7.0)
        axis.set_yticklabels(axis.get_yticklabels(), rotation=90, va="center", fontsize=7.0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xlabel("Clase predicha (% de los píxeles de cada fila)")
        axis.set_ylabel("Clase real")
        axis.set_title(f"Ejecución {spec.run_name}: {spec.description}", loc="center")
        figure.draw_without_rendering()
    return figure


def locate_sample(data_dir: Path, split: str, class_label: int, stem: str) -> SegmentationSample:
    class_dir = data_dir / split / TUMOR_CLASS_NAMES[class_label]
    image_path = class_dir / IMAGES_DIRNAME / f"{stem}.png"
    mask_path = class_dir / MASKS_DIRNAME / image_path.name
    missing_paths = [path for path in (image_path, mask_path) if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Example slice file '{missing_paths[0]}' does not exist.")
    return SegmentationSample(
        image_path=image_path,
        mask_path=mask_path,
        class_label=class_label,
        patient_id=parse_patient_id(image_path),
    )


def load_raw_slice(sample: SegmentationSample) -> Tuple[NDArray[np.generic], NDArray[np.uint8]]:
    image = to_grayscale(read_image_file(sample.image_path, cv2.IMREAD_UNCHANGED))
    mask = read_image_file(sample.mask_path, cv2.IMREAD_GRAYSCALE)
    return image, mask


def compute_zoom_window(mask: NDArray[np.uint8]) -> ZoomWindow:
    tumor_rows, tumor_columns = np.nonzero(mask)
    center_row = int(round((tumor_rows.min() + tumor_rows.max()) / 2.0))
    center_column = int(round((tumor_columns.min() + tumor_columns.max()) / 2.0))
    extent = max(tumor_rows.max() - tumor_rows.min(), tumor_columns.max() - tumor_columns.min())
    half_side = max(int(round(ZOOM_MARGIN_FACTOR * extent / 2.0)), MINIMUM_ZOOM_HALF_SIDE)
    half_side = min(half_side, mask.shape[0] // 2, mask.shape[1] // 2)
    center_row = int(np.clip(center_row, half_side, mask.shape[0] - half_side))
    center_column = int(np.clip(center_column, half_side, mask.shape[1] - half_side))
    return ZoomWindow(
        top=center_row - half_side,
        bottom=center_row + half_side,
        left=center_column - half_side,
        right=center_column + half_side,
    )


def plot_mri_examples(
    examples: Sequence[MriExample],
    slices: Sequence[Tuple[NDArray[np.generic], NDArray[np.uint8]]],
) -> Figure:
    with report_theme():
        figure = Figure(figsize=(TEXT_WIDTH_INCHES, 4.75), layout="constrained")
        axes = figure.subplots(2, len(examples), squeeze=False)
        for column, (example, (image, mask)) in enumerate(zip(examples, slices)):
            slice_axis, detail_axis = axes[:, column]
            display_image = rescale_min_max(image)
            window = compute_zoom_window(mask)
            show_slice(slice_axis, display_image)
            slice_axis.add_patch(
                Rectangle(
                    (window.left, window.top),
                    window.right - window.left,
                    window.bottom - window.top,
                    fill=False,
                    edgecolor=CLOUD_DANCER,
                    linewidth=0.9,
                    linestyle=(0, (3, 2)),
                )
            )
            slice_axis.set_title(example.title, loc="center")
            show_slice(detail_axis, window.crop(display_image))
            outline_region(detail_axis, window.crop(mask) != BACKGROUND_LABEL, VIVA_MAGENTA)
            detail_axis.set_xlabel(example.descriptor, color=TEXT_SECONDARY_COLOR)
        axes[0, 0].set_ylabel("Corte T1-CE completo")
        axes[1, 0].set_ylabel("Detalle de la lesión")
        figure.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=AXIS_COLOR,
                    linewidth=0.9,
                    linestyle=(0, (3, 2)),
                    label="Región ampliada",
                ),
                Line2D(
                    [],
                    [],
                    color=VIVA_MAGENTA,
                    linewidth=CONTOUR_WIDTH,
                    label="Contorno delineado por radiólogos",
                ),
            ],
            loc="outside lower center",
            ncol=2,
        )
        figure.draw_without_rendering()
    return figure


def compute_slice_dice(confusions: NDArray[np.int64], class_label: int) -> NDArray[np.float64]:
    overlap = confusions[:, class_label, class_label].astype(np.float64)
    target_pixels = confusions[:, class_label, :].sum(axis=1).astype(np.float64)
    predicted_pixels = confusions[:, :, class_label].sum(axis=1).astype(np.float64)
    denominator = target_pixels + predicted_pixels
    return np.divide(
        2.0 * overlap, denominator, out=np.full(overlap.shape, np.nan), where=denominator > 0
    )


def select_median_dice_sample(
    samples: Sequence[SegmentationSample], confusions: NDArray[np.int64], class_label: int
) -> SegmentationSample:
    class_indices = np.array(
        [index for index, sample in enumerate(samples) if sample.class_label == class_label]
    )
    slice_dice = compute_slice_dice(confusions[class_indices], class_label)
    distance_to_median = np.abs(slice_dice - np.nanmedian(slice_dice))
    return samples[int(class_indices[int(np.nanargmin(distance_to_median))])]


def preprocess_for_evaluation(
    image: NDArray[np.generic], mask: NDArray[np.uint8]
) -> Tuple[NDArray[np.float32], NDArray[np.uint8]]:
    cropped_image, cropped_mask = crop_background_margins(image, mask)
    resized_image, resized_mask = resize_image_and_mask(
        cropped_image, cropped_mask, DEFAULT_IMAGE_SIZE
    )
    return normalize_intensity(resized_image), resized_mask


def predict_class_mask(checkpoint_path: Path, image: NDArray[np.float32]) -> NDArray[np.uint8]:
    device = torch.device("cpu")
    checkpoint = load_training_checkpoint(checkpoint_path)
    if (checkpoint.image_size, checkpoint.image_size) != image.shape:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' expects {checkpoint.image_size} px slices, "
            f"got {image.shape}."
        )
    model = build_model_from_checkpoint(checkpoint, device)
    batch = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
    batch = batch.unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        class_indices = model(batch).argmax(dim=1).squeeze(0)
    return class_indices.cpu().numpy().astype(np.uint8)


def predict_runs(
    runs: Sequence[TrainingRunSpec],
    runs_dir: Path,
    image: NDArray[np.float32],
    skip_inference: bool,
) -> List[RunPrediction]:
    predictions: List[RunPrediction] = []
    for run in runs:
        checkpoint_path = run.checkpoint_path(runs_dir)
        if skip_inference or not checkpoint_path.is_file():
            predictions.append(RunPrediction(run.name, None))
        else:
            predictions.append(RunPrediction(run.name, predict_class_mask(checkpoint_path, image)))
    return predictions


def compute_class_dice(
    prediction: NDArray[np.uint8], target: NDArray[np.uint8], class_label: int
) -> float:
    predicted_region = prediction == class_label
    target_region = target == class_label
    denominator = int(np.count_nonzero(predicted_region)) + int(np.count_nonzero(target_region))
    if denominator == 0:
        return float("nan")
    return 2.0 * float(np.count_nonzero(predicted_region & target_region)) / float(denominator)


def draw_prediction_panel(
    axis: Axes,
    image: NDArray[np.float32],
    mask: NDArray[np.uint8],
    prediction: RunPrediction,
    class_label: int,
) -> bool:
    if prediction.class_mask is None:
        axis.imshow(np.zeros(image.shape), cmap=ListedColormap([CLOUD_DANCER]), vmin=0.0, vmax=1.0)
        style_image_axis(axis)
        axis.text(
            0.5,
            0.5,
            "Predicción\npendiente",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=SMALL_FONT_SIZE,
            color=TEXT_SECONDARY_COLOR,
            linespacing=1.4,
        )
        axis.set_title(f"Ejecución {prediction.run_name}", loc="center")
        return False

    other_tumor_region = (prediction.class_mask != BACKGROUND_LABEL) & (
        prediction.class_mask != class_label
    )
    show_slice(axis, image)
    fill_region(axis, prediction.class_mask == class_label, VIVA_MAGENTA)
    fill_region(axis, other_tumor_region, CLOUD_DANCER, OTHER_TUMOR_ALPHA)
    outline_region(axis, mask == class_label, CLASSIC_BLUE)
    slice_dice = compute_class_dice(prediction.class_mask, mask, class_label)
    axis.set_title(f"Ejecución {prediction.run_name} (Dice {slice_dice:.2f})", loc="center")
    return bool(other_tumor_region.any())


def plot_segmentation_comparison(
    image: NDArray[np.float32],
    mask: NDArray[np.uint8],
    predictions: Sequence[RunPrediction],
    class_label: int,
) -> Figure:
    class_name = CLASS_DISPLAY_NAMES[class_label].lower()
    with report_theme():
        figure = Figure(figsize=(TEXT_WIDTH_INCHES, 2.0), layout="constrained")
        axes = figure.subplots(1, 2 + len(predictions), squeeze=False)[0]
        show_slice(axes[0], image)
        axes[0].set_title("RM de entrada", loc="center")
        show_slice(axes[1], image)
        fill_region(axes[1], mask == class_label, CLASSIC_BLUE)
        axes[1].set_title("Máscara experta", loc="center")
        shows_other_tumor = [
            draw_prediction_panel(axis, image, mask, prediction, class_label)
            for axis, prediction in zip(axes[2:], predictions)
        ]
        handles: List[Any] = [
            Patch(facecolor=CLASSIC_BLUE, alpha=MASK_ALPHA, label="Máscara experta"),
            Line2D([], [], color=CLASSIC_BLUE, linewidth=CONTOUR_WIDTH, label="Contorno experto"),
            Patch(facecolor=VIVA_MAGENTA, alpha=MASK_ALPHA, label=f"Predicción ({class_name})"),
        ]
        if any(shows_other_tumor):
            handles.append(
                Patch(
                    facecolor=CLOUD_DANCER,
                    edgecolor=AXIS_COLOR,
                    linewidth=0.5,
                    label="Predicción (otro tipo de tumor)",
                )
            )
        figure.legend(handles=handles, loc="outside lower center", ncol=len(handles))
        figure.draw_without_rendering()
    return figure


def build_dataset_split(arguments: argparse.Namespace) -> Dict[str, Figure]:
    subsets = load_dataset_subsets(
        arguments.data_dir, arguments.validation_fraction, arguments.seed
    )
    return {"dataset_split": plot_dataset_split(subsets, measure_tumor_shares(subsets))}


def build_training_curves(arguments: argparse.Namespace) -> Dict[str, Figure]:
    histories = [load_history(run.history_path(arguments.runs_dir)) for run in TRAINING_RUNS]
    constant_learning_rate = load_searched_learning_rate(arguments.runs_dir / SEARCH_RESULT_PATH)
    return {
        "training_curves": plot_training_curves(TRAINING_RUNS, histories, constant_learning_rate)
    }


def build_confusion_matrices(arguments: argparse.Namespace) -> Dict[str, Figure]:
    samples = discover_samples(arguments.data_dir / TESTING_SPLIT)
    figures: Dict[str, Figure] = {}
    for spec in CONFUSION_MATRICES:
        confusions = load_audit_confusions(
            arguments.runs_dir / AUDIT_CONFUSIONS_PATH, spec.run_name
        )
        validate_audit_alignment(confusions, samples)
        figures[spec.stem] = plot_confusion_matrix(normalize_rows(confusions.sum(axis=0)), spec)
    return figures


def build_mri_examples(arguments: argparse.Namespace) -> Dict[str, Figure]:
    slices = [
        load_raw_slice(
            locate_sample(arguments.data_dir, TRAINING_SPLIT, example.class_label, example.stem)
        )
        for example in MRI_EXAMPLES
    ]
    return {"mri_examples": plot_mri_examples(MRI_EXAMPLES, slices)}


def build_segmentation_comparison(arguments: argparse.Namespace) -> Dict[str, Figure]:
    samples = discover_samples(arguments.data_dir / TESTING_SPLIT)
    confusions = load_audit_confusions(arguments.runs_dir / AUDIT_CONFUSIONS_PATH, RUN_E.name)
    validate_audit_alignment(confusions, samples)
    sample = select_median_dice_sample(samples, confusions, GLIOMA_LABEL)
    image, mask = preprocess_for_evaluation(*load_raw_slice(sample))
    predictions = predict_runs(TRAINING_RUNS, arguments.runs_dir, image, arguments.skip_inference)
    pending_runs = [
        prediction.run_name for prediction in predictions if prediction.class_mask is None
    ]
    print(f"Segmentation comparison slice: {sample.image_path.stem} (patient {sample.patient_id}).")
    if pending_runs:
        print(
            "Placeholder panels (no prediction) for runs: "
            + ", ".join(pending_runs)
            + ". Run again with the checkpoints and without --skip-inference."
        )
    return {
        "segmentation_comparison": plot_segmentation_comparison(
            image, mask, predictions, GLIOMA_LABEL
        )
    }


FIGURE_BUILDERS: Final[Mapping[str, Callable[[argparse.Namespace], Dict[str, Figure]]]] = {
    "dataset_split": build_dataset_split,
    "training_curves": build_training_curves,
    "confusion_matrices": build_confusion_matrices,
    "mri_examples": build_mri_examples,
    "segmentation_comparison": build_segmentation_comparison,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--figures", nargs="+", choices=FIGURE_NAMES, default=list(FIGURE_NAMES))
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    for figure_name in arguments.figures:
        for stem, figure in FIGURE_BUILDERS[figure_name](arguments).items():
            print(f"Saved '{save_pdf(figure, arguments.output_dir, stem)}'.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
