"""Unit tests for the equal-budget random search reference."""

import math
import random
from typing import Callable, List, Sequence

import pytest

from src.optimization.fitness import (
    BASELINE_HYPERPARAMETERS,
    SEARCH_SPACE,
    decode_genes,
    encode_hyperparameters,
)
from src.optimization.genetic import GeneticAlgorithmConfig, sample_random_genes
from src.optimization.random_search import (
    DEFAULT_EVALUATIONS,
    RandomSearchConfig,
    RandomSearchRecord,
    run_random_search,
)


def constant_fitness(genes: Sequence[float]) -> float:
    return 0.5


def first_gene_fitness(genes: Sequence[float]) -> float:
    return genes[0]


def build_history(
    config: RandomSearchConfig, fitness_function: Callable[[Sequence[float]], float]
) -> List[RandomSearchRecord]:
    return list(run_random_search(config, fitness_function))


def test_config_rejects_empty_budget() -> None:
    with pytest.raises(ValueError, match="evaluations must be at least 1"):
        RandomSearchConfig(evaluations=0)


def test_config_rejects_zero_progress_every() -> None:
    with pytest.raises(ValueError, match="progress_every must be at least 1"):
        RandomSearchConfig(progress_every=0)


def test_default_budget_matches_genetic_algorithm_defaults() -> None:
    assert DEFAULT_EVALUATIONS == GeneticAlgorithmConfig().max_evaluations


def test_search_runs_exactly_the_requested_budget() -> None:
    history = build_history(RandomSearchConfig(evaluations=7), constant_fitness)

    assert len(history) == 7
    assert [record.evaluations_so_far for record in history] == list(range(1, 8))
    assert [record.evaluation for record in history] == list(range(1, 8))


def test_genes_are_inside_the_unit_hypercube() -> None:
    history = build_history(RandomSearchConfig(evaluations=25), constant_fitness)

    for record in history:
        assert len(record.individual.genes) == len(SEARCH_SPACE)
        assert all(0.0 <= gene <= 1.0 for gene in record.individual.genes)


def test_hyperparameters_are_decoded_by_the_shared_search_space() -> None:
    history = build_history(RandomSearchConfig(evaluations=10), constant_fitness)

    for record in history:
        assert record.individual.hyperparameters == decode_genes(record.individual.genes)


def test_best_so_far_is_monotonic_and_tracks_the_maximum() -> None:
    history = build_history(RandomSearchConfig(evaluations=20), first_gene_fitness)

    seen_fitnesses = [record.individual.fitness for record in history]
    best_so_far = [record.best_so_far.fitness for record in history]

    assert all(later >= earlier for earlier, later in zip(best_so_far, best_so_far[1:]))
    for position, running_best in enumerate(best_so_far):
        assert running_best == max(seen_fitnesses[: position + 1])


def test_same_seed_reproduces_the_same_samples() -> None:
    first = [
        record.individual.genes
        for record in build_history(
            RandomSearchConfig(evaluations=9, seed=7), constant_fitness
        )
    ]
    second = [
        record.individual.genes
        for record in build_history(
            RandomSearchConfig(evaluations=9, seed=7), constant_fitness
        )
    ]

    assert first == second


def test_different_seeds_explore_different_samples() -> None:
    first = [
        record.individual.genes
        for record in build_history(
            RandomSearchConfig(evaluations=9, seed=1), constant_fitness
        )
    ]
    second = [
        record.individual.genes
        for record in build_history(
            RandomSearchConfig(evaluations=9, seed=2), constant_fitness
        )
    ]

    assert first != second


def test_records_report_non_negative_elapsed_time() -> None:
    history = build_history(RandomSearchConfig(evaluations=5), constant_fitness)

    assert all(
        math.isfinite(record.elapsed_seconds) and record.elapsed_seconds >= 0.0
        for record in history
    )


def test_first_sample_is_not_the_seeded_baseline() -> None:
    baseline_genes = tuple(encode_hyperparameters(BASELINE_HYPERPARAMETERS))
    first_random_sample = sample_random_genes(random.Random(RandomSearchConfig().seed))

    assert first_random_sample != baseline_genes
