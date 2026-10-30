"""Tests for the optimization history parsing and convergence figures."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.optimization.reporting import (
    GENETIC_LABEL,
    RANDOM_LABEL,
    SearchProgress,
    build_improvement_table,
    describe_hyperparameters,
    improvement_mask,
    load_genetic_progress,
    load_random_progress,
    plot_generation_fitness,
    plot_search_progress,
)
from src.utils.visualization import save_figure


def build_genetic_payload() -> Dict[str, Any]:
    return {
        "baseline_fitness": 0.44,
        "best_hyperparameters": {
            "learning_rate": 3.15e-4,
            "weight_decay": 1.91e-6,
            "base_filters": 16,
            "depth": 4,
        },
        "history": [
            {
                "generation": 1,
                "fitness_evaluations": 10,
                "best_so_far_fitness": 0.44,
                "population": [{"fitness": 0.2}, {"fitness": 0.44}],
            },
            {
                "generation": 2,
                "fitness_evaluations": 19,
                "best_so_far_fitness": 0.44,
                "population": [{"fitness": 0.3}, {"fitness": 0.44}],
            },
            {
                "generation": 3,
                "fitness_evaluations": 28,
                "best_so_far_fitness": 0.51,
                "population": [{"fitness": 0.35}, {"fitness": 0.51}],
            },
        ],
    }


def build_random_payload() -> Dict[str, Any]:
    return {
        "best_hyperparameters": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "base_filters": 16,
            "depth": 4,
        },
        "history": [
            {"evaluation": index, "evaluations_so_far": index, "best_so_far_fitness": fitness}
            for index, fitness in enumerate([0.1, 0.3, 0.3, 0.45], start=1)
        ],
    }


def write_payload(path: Path, payload: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def genetic_progress(tmp_path: Path) -> SearchProgress:
    return load_genetic_progress(write_payload(tmp_path / "ga.json", build_genetic_payload()))


@pytest.fixture
def random_progress(tmp_path: Path) -> SearchProgress:
    return load_random_progress(
        write_payload(tmp_path / "random.json", build_random_payload())
    )


def test_genetic_progress_parses_history_and_baseline(genetic_progress: SearchProgress) -> None:
    assert genetic_progress.label == GENETIC_LABEL
    assert genetic_progress.evaluations.tolist() == [10, 19, 28]
    assert genetic_progress.best_so_far.tolist() == [0.44, 0.44, 0.51]
    assert genetic_progress.baseline_fitness == pytest.approx(0.44)
    assert genetic_progress.final_fitness == pytest.approx(0.51)
    assert genetic_progress.hyperparameters["base_filters"] == 16
    assert genetic_progress.populations[2] == [0.35, 0.51]


def test_random_progress_uses_per_evaluation_counts(random_progress: SearchProgress) -> None:
    assert random_progress.label == RANDOM_LABEL
    assert random_progress.evaluations.tolist() == [1, 2, 3, 4]
    assert random_progress.baseline_fitness is None
    assert random_progress.improvements.tolist() == [True, True, False, True]


def test_improvement_mask_marks_each_new_maximum() -> None:
    values = [0.2, 0.4, 0.4, 0.35, 0.5]
    assert improvement_mask(__import__("numpy").asarray(values)).tolist() == [
        True,
        True,
        False,
        False,
        True,
    ]


def test_improvement_table_reports_evaluation_and_fitness(
    genetic_progress: SearchProgress,
) -> None:
    assert build_improvement_table(genetic_progress) == [(10, 0.44), (28, 0.51)]


def test_non_monotonic_best_so_far_is_rejected(tmp_path: Path) -> None:
    payload = build_genetic_payload()
    payload["history"][2]["best_so_far_fitness"] = 0.4
    path = write_payload(tmp_path / "broken.json", payload)
    with pytest.raises(ValueError, match="non-decreasing"):
        load_genetic_progress(path)


def test_missing_history_is_rejected(tmp_path: Path) -> None:
    path = write_payload(tmp_path / "empty.json", {"history": []})
    with pytest.raises(ValueError, match="no generation history"):
        load_genetic_progress(path)


def test_describe_hyperparameters_formats_known_fields() -> None:
    text = describe_hyperparameters(
        {"learning_rate": 3.15e-4, "weight_decay": 1.91e-6, "base_filters": 16, "depth": 4}
    )
    assert "3.15e-04" in text and "filters=16" in text and "depth=4" in text
    assert describe_hyperparameters({"learning_rate": 1e-3}) == ""


def test_figures_render_for_both_searches(
    tmp_path: Path, genetic_progress: SearchProgress, random_progress: SearchProgress
) -> None:
    progresses: List[SearchProgress] = [genetic_progress, random_progress]
    figures = {
        "ga_convergence": plot_search_progress(
            [genetic_progress], "GA", "subtitle", "footnote"
        ),
        "budget_comparison": plot_search_progress(progresses, "Budget", "subtitle", "footnote"),
        "ga_generations": plot_generation_fitness(genetic_progress, "Generations", "footnote"),
    }

    for figure in figures.values():
        assert len(figure.axes) == 1
    assert figures["ga_convergence"].axes[0].get_xlim()[1] >= 28
    assert figures["ga_generations"].axes[0].get_xticks().tolist() == [1.0, 2.0, 3.0]
    for stem, figure in figures.items():
        (saved_path,) = save_figure(figure, tmp_path, stem, dpi=40)
        assert saved_path.is_file() and saved_path.stat().st_size > 0
