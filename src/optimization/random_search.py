"""Sample U-Net hyperparameters uniformly as an equal-budget reference for the genetic algorithm.

Unlike the genetic algorithm, no candidate is seeded with the baseline configuration:
the reference curve must reflect pure random sampling over the identical search space.

Usage (imported by the random search script):
    from src.optimization.random_search import RandomSearchConfig, run_random_search
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Final, Iterator, Optional

from .fitness import DEFAULT_SEED
from .genetic import (
    CachedFitnessEvaluator,
    FitnessFunction,
    Individual,
    format_duration,
    format_hyperparameters,
    sample_random_genes,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
DEFAULT_EVALUATIONS: Final[int] = 46
DEFAULT_PROGRESS_EVERY: Final[int] = 5


@dataclass(frozen=True)
class RandomSearchConfig:
    evaluations: int = DEFAULT_EVALUATIONS
    seed: int = DEFAULT_SEED
    progress_every: int = DEFAULT_PROGRESS_EVERY

    def __post_init__(self) -> None:
        if self.evaluations < 1:
            raise ValueError(f"evaluations must be at least 1, got {self.evaluations}.")
        if self.progress_every < 1:
            raise ValueError(f"progress_every must be at least 1, got {self.progress_every}.")


@dataclass(frozen=True)
class RandomSearchRecord:
    evaluation: int
    individual: Individual
    best_so_far: Individual
    evaluations_so_far: int
    elapsed_seconds: float


def describe_evaluation(record: RandomSearchRecord, config: RandomSearchConfig) -> str:
    return (
        f"[Random search] evaluation {record.evaluation}/{config.evaluations} | "
        f"{format_hyperparameters(record.individual.hyperparameters)} | "
        f"fitness {record.individual.fitness:.4f} | "
        f"best so far {record.best_so_far.fitness:.4f} | "
        f"elapsed {format_duration(record.elapsed_seconds)}"
    )


def run_random_search(
    config: RandomSearchConfig, fitness_function: FitnessFunction
) -> Iterator[RandomSearchRecord]:
    rng = random.Random(config.seed)
    evaluator = CachedFitnessEvaluator(fitness_function)
    start_time = time.perf_counter()
    best_so_far: Optional[Individual] = None

    for _ in range(config.evaluations):
        genes = sample_random_genes(rng)
        while evaluator.is_cached(genes):
            genes = sample_random_genes(rng)
        individual = evaluator.evaluate(genes)
        if best_so_far is None or individual.fitness > best_so_far.fitness:
            best_so_far = individual
        record = RandomSearchRecord(
            evaluation=evaluator.evaluation_count,
            individual=individual,
            best_so_far=best_so_far,
            evaluations_so_far=evaluator.evaluation_count,
            elapsed_seconds=time.perf_counter() - start_time,
        )
        if record.evaluation % config.progress_every == 0 or record.evaluation == config.evaluations:
            LOGGER.info(describe_evaluation(record, config))
        yield record
