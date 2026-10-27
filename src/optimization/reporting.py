"""Parse optimization histories and draw convergence figures for the genetic search.

The genetic algorithm stores one record per generation (end-of-generation evaluation
counts), while the random-search reference stores one record per evaluation. Both are
rendered as best-so-far step curves so the two searches can be compared on the same axis:
the number of fitness evaluations, i.e. the compute budget.

Usage (imported by scripts/plot_optimization.py):
    from src.optimization.reporting import load_genetic_progress, load_random_progress
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from ..utils.visualization import (
    OPTIMIZED_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_PRIMARY_COLOR,
    TEXT_SECONDARY_COLOR,
    add_figure_legend,
    add_footnote,
    add_header,
    build_glow_effects,
    create_figure,
    draw_reference_level,
    neon_theme,
)

GENETIC_LABEL: Final[str] = "Genetic algorithm"
RANDOM_LABEL: Final[str] = "Random search"
GA_COLOR: Final[str] = OPTIMIZED_COLOR
RANDOM_COLOR: Final[str] = "#FFC857"
BASELINE_REFERENCE_LABEL: Final[str] = "Baseline configuration"
X_AXIS_LABEL: Final[str] = "Fitness evaluations"
Y_AXIS_LABEL: Final[str] = "Validation Dice (best so far)"
FITNESS_AXIS_LABEL: Final[str] = "Validation Dice (3-epoch fitness)"
GENERATION_AXIS_LABEL: Final[str] = "Generation"
FIGURES_FILENAME_HINT: Final[str] = "figures"


@dataclass(frozen=True)
class SearchProgress:
    label: str
    evaluations: NDArray[np.int64]
    best_so_far: NDArray[np.float64]
    baseline_fitness: Optional[float]
    hyperparameters: Mapping[str, Any]
    populations: Sequence[Sequence[float]] = field(default_factory=tuple)

    @property
    def final_fitness(self) -> float:
        return float(self.best_so_far[-1])

    @property
    def improvements(self) -> NDArray[np.bool_]:
        return improvement_mask(self.best_so_far)

    @property
    def improvement_count(self) -> int:
        return int(self.improvements.sum())


def improvement_mask(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    best_so_far = np.maximum.accumulate(values)
    return np.concatenate(([True], values[1:] > best_so_far[:-1]))


def validate_series(evaluations: NDArray[np.int64], best_so_far: NDArray[np.float64]) -> None:
    if evaluations.size == 0 or best_so_far.size == 0:
        raise ValueError("Optimization history is empty.")
    if evaluations.size != best_so_far.size:
        raise ValueError(
            f"Evaluations and fitness values differ in length "
            f"({evaluations.size} vs {best_so_far.size})."
        )
    if np.any(np.diff(evaluations) <= 0):
        raise ValueError("Evaluation counts must be strictly increasing.")
    if np.any(np.diff(best_so_far) < 0):
        raise ValueError("Best-so-far fitness must be non-decreasing.")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Optimization history not found: '{path}'.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_genetic_progress(path: Path) -> SearchProgress:
    payload = read_json(path)
    records = payload.get("history") or []
    if not records:
        raise ValueError(f"'{path}' has no generation history.")
    evaluations = np.asarray(
        [int(record["fitness_evaluations"]) for record in records], dtype=np.int64
    )
    best_so_far = np.asarray(
        [float(record["best_so_far_fitness"]) for record in records], dtype=np.float64
    )
    validate_series(evaluations, best_so_far)
    populations = [
        [float(individual["fitness"]) for individual in record.get("population", [])]
        for record in records
    ]
    baseline_fitness = payload.get("baseline_fitness")
    return SearchProgress(
        label=GENETIC_LABEL,
        evaluations=evaluations,
        best_so_far=best_so_far,
        baseline_fitness=None if baseline_fitness is None else float(baseline_fitness),
        hyperparameters=dict(payload.get("best_hyperparameters") or {}),
        populations=populations,
    )


def load_random_progress(path: Path) -> SearchProgress:
    payload = read_json(path)
    records = payload.get("history") or []
    if not records:
        raise ValueError(f"'{path}' has no evaluation history.")
    evaluations = np.asarray(
        [int(record["evaluations_so_far"]) for record in records], dtype=np.int64
    )
    best_so_far = np.asarray(
        [float(record["best_so_far_fitness"]) for record in records], dtype=np.float64
    )
    validate_series(evaluations, best_so_far)
    return SearchProgress(
        label=RANDOM_LABEL,
        evaluations=evaluations,
        best_so_far=best_so_far,
        baseline_fitness=None,
        hyperparameters=dict(payload.get("best_hyperparameters") or {}),
    )


def describe_hyperparameters(hyperparameters: Mapping[str, Any]) -> str:
    learning_rate = hyperparameters.get("learning_rate")
    weight_decay = hyperparameters.get("weight_decay")
    base_filters = hyperparameters.get("base_filters")
    depth = hyperparameters.get("depth")
    if None in (learning_rate, weight_decay, base_filters, depth):
        return ""
    return (
        f"lr={float(learning_rate):.2e} · wd={float(weight_decay):.2e} · "
        f"filters={base_filters} · depth={depth}"
    )


def progress_color(label: str) -> str:
    return GA_COLOR if label == GENETIC_LABEL else RANDOM_COLOR


def plot_search_progress(
    progresses: Sequence[SearchProgress],
    title: str,
    subtitle: str,
    footnote: str,
) -> Figure:
    with neon_theme():
        figure = draw_search_progress(progresses, title, subtitle, footnote)
        figure.draw_without_rendering()
    return figure


def draw_search_progress(
    progresses: Sequence[SearchProgress],
    title: str,
    subtitle: str,
    footnote: str,
) -> Figure:
    figure, (axis,) = create_figure((11.0, 5.8), columns=1)
    handles: List[Line2D] = []
    for progress in progresses:
        color = progress_color(progress.label)
        line = axis.step(
            progress.evaluations,
            progress.best_so_far,
            where="post",
            color=color,
            linewidth=3.2,
            label=progress.label,
        )[0]
        line.set_path_effects(build_glow_effects(color, 3.2))
        improvements = progress.improvements
        axis.scatter(
            progress.evaluations[improvements],
            progress.best_so_far[improvements],
            s=46,
            color=color,
            edgecolor="#0B0F1A",
            linewidth=1.4,
            zorder=5,
        )
        axis.annotate(
            f"{progress.final_fitness:.4f}",
            xy=(progress.evaluations[-1], progress.final_fitness),
            xytext=(-10, 10),
            textcoords="offset points",
            ha="right",
            color=TEXT_PRIMARY_COLOR,
            fontsize=11,
            fontweight="bold",
        )
        handles.append(Line2D([], [], color=color, linewidth=3.2, label=progress.label))

    reference = next(
        (
            progress.baseline_fitness
            for progress in progresses
            if progress.baseline_fitness is not None
        ),
        None,
    )
    if reference is not None:
        draw_reference_level(axis, reference, BASELINE_REFERENCE_LABEL)
    axis.set_xlabel(X_AXIS_LABEL, color=TEXT_SECONDARY_COLOR)
    axis.set_ylabel(Y_AXIS_LABEL, color=TEXT_SECONDARY_COLOR)
    add_header(figure, title, subtitle)
    add_figure_legend(figure, handles)
    add_footnote(figure, footnote)
    return figure


def plot_generation_fitness(progress: SearchProgress, title: str, footnote: str) -> Figure:
    with neon_theme():
        figure = draw_generation_fitness(progress, title, footnote)
        figure.draw_without_rendering()
    return figure


def draw_generation_fitness(progress: SearchProgress, title: str, footnote: str) -> Figure:
    populations = [values for values in progress.populations if values]
    if not populations:
        raise ValueError("The genetic history has no per-generation populations to plot.")
    figure, (axis,) = create_figure((11.0, 5.8), columns=1)
    positions = np.arange(1, len(populations) + 1)
    boxplot = axis.boxplot(
        populations,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": TEXT_PRIMARY_COLOR, "linewidth": 2.0},
        whiskerprops={"color": TEXT_MUTED_COLOR},
        capprops={"color": TEXT_MUTED_COLOR},
        flierprops={"markeredgecolor": TEXT_MUTED_COLOR, "markerfacecolor": "none"},
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor("#2A1830")
        patch.set_edgecolor(GA_COLOR)
        patch.set_linewidth(1.6)
    axis.step(
        positions,
        progress.best_so_far,
        where="post",
        color=GA_COLOR,
        linewidth=3.2,
        label="Best so far",
        zorder=4,
    )
    axis.scatter(
        positions, [max(values) for values in populations], s=42, color=GA_COLOR, zorder=5
    )
    if progress.baseline_fitness is not None:
        draw_reference_level(axis, progress.baseline_fitness, BASELINE_REFERENCE_LABEL)
    axis.set_xticks(positions)
    axis.set_xlabel(GENERATION_AXIS_LABEL, color=TEXT_SECONDARY_COLOR)
    axis.set_ylabel(FITNESS_AXIS_LABEL, color=TEXT_SECONDARY_COLOR)
    axis.set_title(
        "Population fitness by generation (box = distribution, dot = best individual)",
        color=TEXT_SECONDARY_COLOR,
        fontsize=12.5,
        loc="left",
    )
    add_header(
        figure,
        title,
        f"{len(populations)} generations · {int(progress.evaluations[-1])} evaluations",
    )
    add_footnote(figure, footnote)
    return figure


def build_improvement_table(progress: SearchProgress) -> List[Tuple[int, float]]:
    return [
        (int(evaluation), float(fitness))
        for evaluation, fitness, improved in zip(
            progress.evaluations, progress.best_so_far, progress.improvements
        )
        if improved
    ]
