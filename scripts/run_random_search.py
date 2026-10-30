"""Run the equal-budget uniform random search that serves as a reference for the genetic algorithm.

Usage (from the project root):
    python -m scripts.run_random_search --evaluations 67 [--device auto|cpu|cuda] [--amp] [--output-dir runs/random_search]
"""

import argparse
import logging
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, List, Sequence

from scripts.run_optimization import (
    build_shared_dataloaders,
    configure_logging,
    describe_gene,
    write_json_atomically,
)
from src.optimization.fitness import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FITNESS_EPOCHS,
    DEFAULT_SEED,
    SEARCH_SPACE,
    FitnessSettings,
    evaluate_fitness,
)
from src.optimization.genetic import Individual, format_duration
from src.optimization.random_search import (
    DEFAULT_EVALUATIONS,
    DEFAULT_PROGRESS_EVERY,
    RandomSearchConfig,
    RandomSearchRecord,
    run_random_search,
)
from src.utils.training import resolve_amp_mode, resolve_device

DEFAULT_OUTPUT_DIR: Final[Path] = Path("runs/random_search")
RESULT_FILENAME: Final[str] = "random_search_results.json"
BUDGET_NOTE: Final[str] = (
    "Fair comparison: pass --evaluations equal to the 'fitness_evaluations' value "
    "stored in runs/optimization/best_hyperparameters.json."
)
LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--evaluations",
        type=int,
        default=DEFAULT_EVALUATIONS,
        help=f"Number of fitness evaluations (default {DEFAULT_EVALUATIONS}). {BUDGET_NOTE}",
    )
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_FITNESS_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_random_search_config(arguments: argparse.Namespace) -> RandomSearchConfig:
    return RandomSearchConfig(
        evaluations=arguments.evaluations,
        seed=arguments.seed,
        progress_every=arguments.progress_every,
    )


def build_fitness_settings(arguments: argparse.Namespace) -> FitnessSettings:
    return FitnessSettings(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        max_train_batches=arguments.limit_batches,
        max_validation_batches=arguments.limit_batches,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
        device=arguments.device,
        amp=arguments.amp,
    )


def log_run_header(
    search_config: RandomSearchConfig, fitness_settings: FitnessSettings, result_path: Path
) -> None:
    device = resolve_device(fitness_settings.device)
    amp_enabled = resolve_amp_mode(fitness_settings.amp, device)
    image_height, image_width = fitness_settings.image_size
    LOGGER.info(
        f"Random search | evaluations {search_config.evaluations} | "
        f"seed {search_config.seed} | no baseline seeding"
    )
    LOGGER.info(
        f"Fitness | {fitness_settings.epochs} epochs | batch {fitness_settings.batch_size} | "
        f"image {image_height}x{image_width} | device {device} | "
        f"AMP {'on' if amp_enabled else 'off'} | workers {fitness_settings.num_workers} | "
        f"batch limit {fitness_settings.max_train_batches or 'none'}"
    )
    LOGGER.info("Search space | " + " | ".join(describe_gene(gene) for gene in SEARCH_SPACE))
    LOGGER.info(
        f"Progress is saved to '{result_path}' every {search_config.progress_every} evaluations."
    )
    LOGGER.info(BUDGET_NOTE)


def serialize_individual(individual: Individual) -> Dict[str, Any]:
    return {
        "hyperparameters": asdict(individual.hyperparameters),
        "fitness": individual.fitness,
        "evaluation_seconds": round(individual.evaluation_seconds, 3),
        "genes": list(individual.genes),
    }


def serialize_record(record: RandomSearchRecord) -> Dict[str, Any]:
    return {
        "evaluation": record.evaluation,
        "hyperparameters": asdict(record.individual.hyperparameters),
        "fitness": record.individual.fitness,
        "best_so_far_fitness": record.best_so_far.fitness,
        "evaluations_so_far": record.evaluations_so_far,
        "elapsed_seconds": round(record.elapsed_seconds, 3),
    }


def build_result_payload(
    history: Sequence[RandomSearchRecord],
    search_config: RandomSearchConfig,
    fitness_settings: FitnessSettings,
) -> Dict[str, Any]:
    latest_record = history[-1]
    best_individual = latest_record.best_so_far
    return {
        "best_hyperparameters": asdict(best_individual.hyperparameters),
        "best_fitness": best_individual.fitness,
        "best_genes": list(best_individual.genes),
        "best_individual": serialize_individual(best_individual),
        "evaluations": latest_record.evaluations_so_far,
        "elapsed_seconds": round(latest_record.elapsed_seconds, 3),
        "random_search": asdict(search_config),
        "fitness_settings": asdict(fitness_settings),
        "history": [serialize_record(record) for record in history],
    }


def log_final_summary(
    history: Sequence[RandomSearchRecord], result_path: Path
) -> None:
    final_record = history[-1]
    LOGGER.info(
        f"Random search finished in {format_duration(final_record.elapsed_seconds)} "
        f"with {final_record.evaluations_so_far} fitness evaluations."
    )
    LOGGER.info(
        f"Best fitness (validation Dice): {final_record.best_so_far.fitness:.4f} | "
        f"{final_record.best_so_far.hyperparameters}"
    )
    LOGGER.info(f"Reference results saved to '{result_path}'.")
    LOGGER.info(
        "Compare 'best_so_far_fitness' against the GA history in "
        "runs/optimization/best_hyperparameters.json."
    )


def main() -> None:
    arguments = parse_arguments()
    configure_logging()
    search_config = build_random_search_config(arguments)
    fitness_settings = build_fitness_settings(arguments)
    result_path: Path = arguments.output_dir / RESULT_FILENAME
    log_run_header(search_config, fitness_settings, result_path)

    fitness_function = partial(
        evaluate_fitness,
        settings=fitness_settings,
        dataloaders=build_shared_dataloaders(fitness_settings),
    )
    history: List[RandomSearchRecord] = []
    for record in run_random_search(search_config, fitness_function):
        history.append(record)
        reached_checkpoint = record.evaluation % search_config.progress_every == 0
        finished = record.evaluation == search_config.evaluations
        if reached_checkpoint or finished:
            write_json_atomically(
                result_path, build_result_payload(history, search_config, fitness_settings)
            )
    log_final_summary(history, result_path)


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(f"Error: {error}")
