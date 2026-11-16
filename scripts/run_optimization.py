"""Run the genetic algorithm that tunes U-Net hyperparameters and save the best configuration to JSON.

Usage (from the project root):
    python -m scripts.run_optimization --population-size 10 --generations 5 [--device auto|cpu|cuda] [--amp] [--output-dir runs/optimization]
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Sequence

from src.dataset.dataloader import SegmentationDataLoaders
from src.optimization.fitness import (
    BASELINE_HYPERPARAMETERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FITNESS_EPOCHS,
    DEFAULT_SEED,
    SEARCH_SPACE,
    FitnessSettings,
    Gene,
    Hyperparameters,
    LogUniformGene,
    build_fitness_dataloaders,
    encode_hyperparameters,
    evaluate_fitness,
)
from src.optimization.genetic import (
    DEFAULT_BLEND_ALPHA,
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_ELITE_SIZE,
    DEFAULT_GENERATIONS,
    DEFAULT_MUTATION_RATE,
    DEFAULT_MUTATION_SIGMA,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_TOURNAMENT_SIZE,
    GenerationRecord,
    GeneticAlgorithmConfig,
    Individual,
    evolve,
    format_duration,
    format_hyperparameters,
)
from src.utils.training import resolve_amp_mode, resolve_device

DEFAULT_OUTPUT_DIR: Final[Path] = Path("runs/optimization")
RESULT_FILENAME: Final[str] = "best_hyperparameters.json"
FINAL_TRAINING_EPOCHS: Final[int] = 40
FINAL_TRAINING_OUTPUT_DIR: Final[str] = "runs/optimized"
LOG_FORMAT: Final[str] = "%(asctime)s | %(message)s"
LOG_DATE_FORMAT: Final[str] = "%H:%M:%S"
LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--crossover-rate", type=float, default=DEFAULT_CROSSOVER_RATE)
    parser.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE)
    parser.add_argument("--mutation-sigma", type=float, default=DEFAULT_MUTATION_SIGMA)
    parser.add_argument("--blend-alpha", type=float, default=DEFAULT_BLEND_ALPHA)
    parser.add_argument("--tournament-size", type=int, default=DEFAULT_TOURNAMENT_SIZE)
    parser.add_argument("--elite-size", type=int, default=DEFAULT_ELITE_SIZE)
    parser.add_argument("--include-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_FITNESS_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )


def build_genetic_config(arguments: argparse.Namespace) -> GeneticAlgorithmConfig:
    return GeneticAlgorithmConfig(
        population_size=arguments.population_size,
        generations=arguments.generations,
        crossover_rate=arguments.crossover_rate,
        mutation_rate=arguments.mutation_rate,
        mutation_sigma=arguments.mutation_sigma,
        blend_alpha=arguments.blend_alpha,
        tournament_size=arguments.tournament_size,
        elite_size=arguments.elite_size,
        include_baseline=arguments.include_baseline,
        seed=arguments.seed,
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


def build_shared_dataloaders(settings: FitnessSettings) -> Optional[SegmentationDataLoaders]:
    persistent_workers_change_augmentations_between_evaluations = settings.num_workers > 0
    if persistent_workers_change_augmentations_between_evaluations:
        return None
    return build_fitness_dataloaders(settings)


def describe_gene(gene: Gene) -> str:
    if isinstance(gene, LogUniformGene):
        return f"{gene.name} log-uniform [{gene.lower_bound:.0e}, {gene.upper_bound:.0e}]"
    return f"{gene.name} {{{', '.join(str(choice) for choice in gene.choices)}}}"


def log_run_header(
    genetic_config: GeneticAlgorithmConfig, fitness_settings: FitnessSettings, result_path: Path
) -> None:
    device = resolve_device(fitness_settings.device)
    amp_enabled = resolve_amp_mode(fitness_settings.amp, device)
    image_height, image_width = fitness_settings.image_size
    LOGGER.info(
        f"Genetic algorithm | population {genetic_config.population_size} | "
        f"generations {genetic_config.generations} | elite {genetic_config.elite_size} | "
        f"tournament {genetic_config.tournament_size} | "
        f"crossover {genetic_config.crossover_rate} (BLX-alpha {genetic_config.blend_alpha}) | "
        f"mutation {genetic_config.mutation_rate} (sigma {genetic_config.mutation_sigma}) | "
        f"baseline seeded {'yes' if genetic_config.include_baseline else 'no'} | "
        f"seed {genetic_config.seed}"
    )
    LOGGER.info(
        f"Fitness | {fitness_settings.epochs} epochs | batch {fitness_settings.batch_size} | "
        f"image {image_height}x{image_width} | device {device} | "
        f"AMP {'on' if amp_enabled else 'off'} | workers {fitness_settings.num_workers} | "
        f"batch limit {fitness_settings.max_train_batches or 'none'}"
    )
    LOGGER.info("Search space | " + " | ".join(describe_gene(gene) for gene in SEARCH_SPACE))
    LOGGER.info(
        f"Budget: at most {genetic_config.max_evaluations} fitness evaluations "
        f"(repeated hyperparameters reuse their cached score)."
    )
    LOGGER.info(f"Progress is saved to '{result_path}' after every generation.")


def find_baseline_fitness(first_generation: GenerationRecord) -> Optional[float]:
    baseline_genes = tuple(encode_hyperparameters(BASELINE_HYPERPARAMETERS))
    return next(
        (
            individual.fitness
            for individual in first_generation.population
            if individual.genes == baseline_genes
        ),
        None,
    )


def serialize_individual(individual: Individual) -> Dict[str, Any]:
    return {
        "hyperparameters": asdict(individual.hyperparameters),
        "fitness": individual.fitness,
        "origin": individual.origin,
        "evaluation_seconds": round(individual.evaluation_seconds, 3),
        "genes": list(individual.genes),
    }


def serialize_generation(record: GenerationRecord) -> Dict[str, Any]:
    return {
        "generation": record.generation,
        "best_fitness": record.best.fitness,
        "mean_fitness": record.mean_fitness,
        "best_so_far_fitness": record.best_so_far.fitness,
        "fitness_evaluations": record.evaluations_so_far,
        "elapsed_seconds": round(record.elapsed_seconds, 3),
        "population": [serialize_individual(individual) for individual in record.population],
    }


def build_result_payload(
    history: Sequence[GenerationRecord],
    genetic_config: GeneticAlgorithmConfig,
    fitness_settings: FitnessSettings,
) -> Dict[str, Any]:
    latest_record = history[-1]
    best_individual = latest_record.best_so_far
    return {
        "best_hyperparameters": asdict(best_individual.hyperparameters),
        "best_fitness": best_individual.fitness,
        "baseline_fitness": find_baseline_fitness(history[0]),
        "best_genes": list(best_individual.genes),
        "completed_generations": latest_record.generation,
        "total_generations": genetic_config.generations,
        "fitness_evaluations": latest_record.evaluations_so_far,
        "elapsed_seconds": round(latest_record.elapsed_seconds, 3),
        "genetic_algorithm": asdict(genetic_config),
        "fitness_settings": asdict(fitness_settings),
        "history": [serialize_generation(record) for record in history],
    }


def write_json_atomically(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def build_final_training_command(
    hyperparameters: Hyperparameters, fitness_settings: FitnessSettings
) -> str:
    amp_flag = " --amp" if fitness_settings.amp else ""
    return (
        f"python -m scripts.train_baseline --epochs {FINAL_TRAINING_EPOCHS} "
        f"--learning-rate {hyperparameters.learning_rate:.6g} "
        f"--weight-decay {hyperparameters.weight_decay:.6g} "
        f"--base-filters {hyperparameters.base_filters} --depth {hyperparameters.depth} "
        f"--batch-size {fitness_settings.batch_size} --seed {fitness_settings.seed} "
        f"--device auto{amp_flag} --output-dir {FINAL_TRAINING_OUTPUT_DIR}"
    )


def log_final_summary(
    history: Sequence[GenerationRecord], result_path: Path, fitness_settings: FitnessSettings
) -> None:
    final_record = history[-1]
    best_individual = final_record.best_so_far
    baseline_fitness = find_baseline_fitness(history[0])
    LOGGER.info(
        f"Optimization finished in {format_duration(final_record.elapsed_seconds)} "
        f"with {final_record.evaluations_so_far} fitness evaluations."
    )
    LOGGER.info(
        f"Best fitness (validation Dice): {best_individual.fitness:.4f} | "
        f"{format_hyperparameters(best_individual.hyperparameters)}"
    )
    if baseline_fitness is not None:
        LOGGER.info(
            f"Baseline fitness: {baseline_fitness:.4f} | "
            f"improvement {best_individual.fitness - baseline_fitness:+.4f}"
        )
    LOGGER.info(f"Best hyperparameters saved to '{result_path}'.")
    LOGGER.info(
        "Train the optimized model with: "
        f"{build_final_training_command(best_individual.hyperparameters, fitness_settings)}"
    )


def main() -> None:
    arguments = parse_arguments()
    configure_logging()
    genetic_config = build_genetic_config(arguments)
    fitness_settings = build_fitness_settings(arguments)
    result_path: Path = arguments.output_dir / RESULT_FILENAME
    log_run_header(genetic_config, fitness_settings, result_path)

    fitness_function = partial(
        evaluate_fitness,
        settings=fitness_settings,
        dataloaders=build_shared_dataloaders(fitness_settings),
    )
    history: List[GenerationRecord] = []
    for record in evolve(genetic_config, fitness_function):
        history.append(record)
        write_json_atomically(
            result_path, build_result_payload(history, genetic_config, fitness_settings)
        )
    log_final_summary(history, result_path, fitness_settings)


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(f"Error: {error}")
