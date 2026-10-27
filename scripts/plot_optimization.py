"""Render the Phase 2 optimization figures: GA convergence and budget comparison.

Figures (in ``<output-dir>/figures``):
    ga_convergence.png            best-so-far fitness of the genetic search per evaluation
    ga_fitness_by_generation.png  population fitness distribution and best-so-far per generation
    budget_comparison.png         GA vs equal-budget random search (skipped until the random
                                  search JSON produced by scripts/run_random_search.py exists)

Usage (from the project root):
    python -m scripts.plot_optimization [--ga-history runs/optimization/best_hyperparameters.json]
        [--random-search runs/random_search/random_search_results.json]
        [--output-dir runs/optimization] [--formats png pdf] [--dpi 200]
"""

import argparse
from pathlib import Path
from typing import Final, List, Sequence

from matplotlib.figure import Figure

from src.optimization.reporting import (
    GENETIC_LABEL,
    RANDOM_LABEL,
    SearchProgress,
    build_improvement_table,
    describe_hyperparameters,
    load_genetic_progress,
    load_random_progress,
    plot_generation_fitness,
    plot_search_progress,
)
from src.utils.visualization import DEFAULT_DPI, save_figure

DEFAULT_GA_HISTORY: Final[Path] = Path("runs/optimization/best_hyperparameters.json")
DEFAULT_RANDOM_SEARCH: Final[Path] = Path("runs/random_search/random_search_results.json")
DEFAULT_OUTPUT_DIR: Final[Path] = Path("runs/optimization")
FIGURES_DIRNAME: Final[str] = "figures"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ga-history", type=Path, default=DEFAULT_GA_HISTORY)
    parser.add_argument("--random-search", type=Path, default=DEFAULT_RANDOM_SEARCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png"])
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser.parse_args()


def export_figure(
    figure: Figure, output_dir: Path, stem: str, formats: Sequence[str], dpi: int
) -> List[Path]:
    saved_paths = save_figure(figure, output_dir, stem, formats, dpi)
    for path in saved_paths:
        print(f"Saved '{path}'.")
    return saved_paths


def describe_progress(progress: SearchProgress) -> str:
    return (
        f"{progress.label:<18} | evaluations {int(progress.evaluations[-1]):>3} | "
        f"best fitness {progress.final_fitness:.4f} | "
        f"improvements {progress.improvement_count} | "
        f"{describe_hyperparameters(progress.hyperparameters)}"
    )


def describe_generations(progress: SearchProgress) -> str:
    generations = len(progress.evaluations)
    population_size = len(progress.populations[0]) if progress.populations else 0
    return f"{generations} generations · population {population_size}"


def print_summary(progresses: Sequence[SearchProgress]) -> None:
    baseline = next(
        (progress.baseline_fitness for progress in progresses if progress.baseline_fitness),
        None,
    )
    for progress in progresses:
        header = progress.label
        print(f"\n{header}")
        print("-" * len(header))
        print(describe_progress(progress))
        for evaluation, fitness in build_improvement_table(progress):
            print(f"  evaluation {evaluation:>3}: best-so-far {fitness:.4f}")
        if baseline is not None:
            print(
                f"  vs baseline seed {baseline:.4f}: "
                f"{progress.final_fitness - baseline:+.4f}"
            )


def main() -> None:
    arguments = parse_arguments()
    figures_dir = arguments.output_dir / FIGURES_DIRNAME
    genetic = load_genetic_progress(arguments.ga_history)
    progresses: List[SearchProgress] = [genetic]
    random_search = (
        load_random_progress(arguments.random_search)
        if arguments.random_search.is_file()
        else None
    )
    if random_search is not None:
        progresses.append(random_search)
    print_summary(progresses)

    hyperparameters = describe_hyperparameters(genetic.hyperparameters)
    export_figure(
        plot_search_progress(
            [genetic],
            "Genetic algorithm convergence",
            f"{describe_generations(genetic)} · {int(genetic.evaluations[-1])} evaluations",
            f"Best configuration: {hyperparameters}.",
        ),
        figures_dir,
        "ga_convergence",
        arguments.formats,
        arguments.dpi,
    )
    export_figure(
        plot_generation_fitness(
            genetic,
            "Genetic algorithm fitness by generation",
            "Fitness is the validation Dice after 3 training epochs, the GA's cheap proxy; "
            "best-so-far includes the seeded baseline configuration.",
        ),
        figures_dir,
        "ga_fitness_by_generation",
        arguments.formats,
        arguments.dpi,
    )

    if random_search is None:
        print(
            f"\nSkipping the budget comparison: '{arguments.random_search}' not found. "
            "Run 'python -m scripts.run_random_search' first."
        )
        return
    budget_footnote = (
        f"GA {int(genetic.evaluations[-1])} evaluations · "
        f"random search {int(random_search.evaluations[-1])} evaluations"
    )
    export_figure(
        plot_search_progress(
            progresses,
            "Genetic algorithm vs random search",
            "Equal compute budget: same number of fitness evaluations",
            budget_footnote,
        ),
        figures_dir,
        "budget_comparison",
        arguments.formats,
        arguments.dpi,
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
