"""Evolve U-Net hyperparameters with tournament selection, BLX-alpha crossover, Gaussian mutation and elitism.

Usage (imported by the optimization script):
    from src.optimization.genetic import GeneticAlgorithmConfig, evolve
"""

import logging
import random
import statistics
import time
from dataclasses import dataclass, replace
from typing import (
    AbstractSet,
    Callable,
    Dict,
    Final,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

from .fitness import (
    BASELINE_HYPERPARAMETERS,
    DEFAULT_SEED,
    SEARCH_SPACE,
    Hyperparameters,
    decode_genes,
    encode_hyperparameters,
)

Genes = Tuple[float, ...]
FitnessFunction = Callable[[Sequence[float]], float]
IndividualOrigin = Literal["evaluated", "cached", "elite"]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
SECONDS_PER_MINUTE: Final[int] = 60
SECONDS_PER_HOUR: Final[int] = 3600
DEFAULT_POPULATION_SIZE: Final[int] = 10
DEFAULT_GENERATIONS: Final[int] = 5
DEFAULT_CROSSOVER_RATE: Final[float] = 0.8
DEFAULT_MUTATION_RATE: Final[float] = 0.25
DEFAULT_MUTATION_SIGMA: Final[float] = 0.2
DEFAULT_BLEND_ALPHA: Final[float] = 0.5
DEFAULT_TOURNAMENT_SIZE: Final[int] = 3
DEFAULT_ELITE_SIZE: Final[int] = 1
MAX_NOVELTY_MUTATIONS: Final[int] = 20


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    population_size: int = DEFAULT_POPULATION_SIZE
    generations: int = DEFAULT_GENERATIONS
    crossover_rate: float = DEFAULT_CROSSOVER_RATE
    mutation_rate: float = DEFAULT_MUTATION_RATE
    mutation_sigma: float = DEFAULT_MUTATION_SIGMA
    blend_alpha: float = DEFAULT_BLEND_ALPHA
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE
    elite_size: int = DEFAULT_ELITE_SIZE
    include_baseline: bool = True
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError(f"population_size must be at least 2, got {self.population_size}.")
        if self.generations < 1:
            raise ValueError(f"generations must be at least 1, got {self.generations}.")
        for rate_name, rate in (
            ("crossover_rate", self.crossover_rate),
            ("mutation_rate", self.mutation_rate),
        ):
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{rate_name} must be in [0, 1], got {rate}.")
        if self.mutation_sigma <= 0.0:
            raise ValueError(f"mutation_sigma must be positive, got {self.mutation_sigma}.")
        if self.blend_alpha < 0.0:
            raise ValueError(f"blend_alpha must be non-negative, got {self.blend_alpha}.")
        if self.tournament_size < 1:
            raise ValueError(f"tournament_size must be at least 1, got {self.tournament_size}.")
        if not 0 <= self.elite_size < self.population_size:
            raise ValueError(
                f"elite_size must be between 0 and population_size - 1 ({self.population_size - 1}), "
                f"got {self.elite_size}."
            )

    @property
    def offspring_per_generation(self) -> int:
        return self.population_size - self.elite_size

    @property
    def max_evaluations(self) -> int:
        return self.population_size + (self.generations - 1) * self.offspring_per_generation


@dataclass(frozen=True)
class Individual:
    genes: Genes
    hyperparameters: Hyperparameters
    fitness: float
    evaluation_seconds: float
    origin: IndividualOrigin


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    population: Tuple[Individual, ...]
    best_so_far: Individual
    evaluations_so_far: int
    elapsed_seconds: float

    @property
    def best(self) -> Individual:
        return self.population[0]

    @property
    def mean_fitness(self) -> float:
        return statistics.fmean(individual.fitness for individual in self.population)


class CachedFitnessEvaluator:
    def __init__(self, fitness_function: FitnessFunction) -> None:
        self.fitness_function = fitness_function
        self.fitness_by_hyperparameters: Dict[Hyperparameters, float] = {}

    @property
    def evaluation_count(self) -> int:
        return len(self.fitness_by_hyperparameters)

    @property
    def known_hyperparameters(self) -> AbstractSet[Hyperparameters]:
        return self.fitness_by_hyperparameters.keys()

    def is_cached(self, genes: Genes) -> bool:
        return decode_genes(genes) in self.fitness_by_hyperparameters

    def evaluate(self, genes: Genes) -> Individual:
        hyperparameters = decode_genes(genes)
        if hyperparameters in self.fitness_by_hyperparameters:
            cached_fitness = self.fitness_by_hyperparameters[hyperparameters]
            return Individual(genes, hyperparameters, cached_fitness, 0.0, "cached")
        start_time = time.perf_counter()
        fitness = self.fitness_function(genes)
        evaluation_seconds = time.perf_counter() - start_time
        self.fitness_by_hyperparameters[hyperparameters] = fitness
        return Individual(genes, hyperparameters, fitness, evaluation_seconds, "evaluated")


def format_duration(total_seconds: float) -> str:
    hours, remaining_seconds = divmod(round(total_seconds), SECONDS_PER_HOUR)
    minutes, seconds = divmod(remaining_seconds, SECONDS_PER_MINUTE)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def format_hyperparameters(hyperparameters: Hyperparameters) -> str:
    return (
        f"lr={hyperparameters.learning_rate:.2e} wd={hyperparameters.weight_decay:.2e} "
        f"filters={hyperparameters.base_filters} depth={hyperparameters.depth}"
    )


def reflect_into_unit_interval(value: float) -> float:
    folded_value = abs(value) % 2.0
    return 2.0 - folded_value if folded_value > 1.0 else folded_value


def sample_random_genes(rng: random.Random) -> Genes:
    return tuple(rng.random() for _ in SEARCH_SPACE)


def initial_population_genes(config: GeneticAlgorithmConfig, rng: random.Random) -> List[Genes]:
    seeded_genes: List[Genes] = (
        [tuple(encode_hyperparameters(BASELINE_HYPERPARAMETERS))] if config.include_baseline else []
    )
    random_genes = [
        sample_random_genes(rng) for _ in range(config.population_size - len(seeded_genes))
    ]
    return seeded_genes + random_genes


def tournament_select(
    population: Sequence[Individual], tournament_size: int, rng: random.Random
) -> Individual:
    contenders = rng.sample(list(population), min(tournament_size, len(population)))
    return max(contenders, key=lambda individual: individual.fitness)


def blend_genes(
    first_parent: Genes, second_parent: Genes, alpha: float, rng: random.Random
) -> Genes:
    child_genes: List[float] = []
    for first_gene, second_gene in zip(first_parent, second_parent):
        lower_gene, upper_gene = min(first_gene, second_gene), max(first_gene, second_gene)
        spread = upper_gene - lower_gene
        blended_gene = rng.uniform(lower_gene - alpha * spread, upper_gene + alpha * spread)
        child_genes.append(reflect_into_unit_interval(blended_gene))
    return tuple(child_genes)


def gaussian_mutation(
    genes: Genes, mutation_rate: float, sigma: float, rng: random.Random
) -> Genes:
    return tuple(
        reflect_into_unit_interval(gene + rng.gauss(0.0, sigma))
        if rng.random() < mutation_rate
        else gene
        for gene in genes
    )


def mutate_until_novel(
    genes: Genes,
    claimed_hyperparameters: AbstractSet[Hyperparameters],
    config: GeneticAlgorithmConfig,
    rng: random.Random,
) -> Genes:
    for _ in range(MAX_NOVELTY_MUTATIONS):
        if decode_genes(genes) not in claimed_hyperparameters:
            return genes
        genes = gaussian_mutation(genes, mutation_rate=1.0, sigma=config.mutation_sigma, rng=rng)
    return genes


def breed_offspring(
    population: Sequence[Individual],
    offspring_count: int,
    known_hyperparameters: AbstractSet[Hyperparameters],
    config: GeneticAlgorithmConfig,
    rng: random.Random,
) -> List[Genes]:
    claimed_hyperparameters = set(known_hyperparameters)
    offspring: List[Genes] = []
    while len(offspring) < offspring_count:
        first_parent = tournament_select(population, config.tournament_size, rng).genes
        second_parent = tournament_select(population, config.tournament_size, rng).genes
        if rng.random() < config.crossover_rate:
            children = (
                blend_genes(first_parent, second_parent, config.blend_alpha, rng),
                blend_genes(first_parent, second_parent, config.blend_alpha, rng),
            )
        else:
            children = (first_parent, second_parent)
        for child in children:
            mutated_child = gaussian_mutation(
                child, config.mutation_rate, config.mutation_sigma, rng
            )
            novel_child = mutate_until_novel(mutated_child, claimed_hyperparameters, config, rng)
            claimed_hyperparameters.add(decode_genes(novel_child))
            offspring.append(novel_child)
    return offspring[:offspring_count]


def rank_by_fitness(individuals: Sequence[Individual]) -> Tuple[Individual, ...]:
    return tuple(sorted(individuals, key=lambda individual: individual.fitness, reverse=True))


def carry_over_elites(ranked_population: Sequence[Individual], elite_size: int) -> List[Individual]:
    return [
        replace(individual, origin="elite", evaluation_seconds=0.0)
        for individual in ranked_population[:elite_size]
    ]


def describe_evaluation(individual: Individual) -> str:
    outcome = (
        "cached, no training"
        if individual.origin == "cached"
        else f"trained in {format_duration(individual.evaluation_seconds)}"
    )
    return (
        f"{format_hyperparameters(individual.hyperparameters)} | "
        f"fitness {individual.fitness:.4f} | {outcome}"
    )


def evaluate_candidates(
    candidate_genes: Sequence[Genes],
    evaluator: CachedFitnessEvaluator,
    generation: int,
    config: GeneticAlgorithmConfig,
) -> List[Individual]:
    individuals: List[Individual] = []
    for candidate_number, genes in enumerate(candidate_genes, start=1):
        label = (
            f"[Gen {generation}/{config.generations}] "
            f"individual {candidate_number}/{len(candidate_genes)}"
        )
        if not evaluator.is_cached(genes):
            LOGGER.info(f"{label} | training {format_hyperparameters(decode_genes(genes))} ...")
        individual = evaluator.evaluate(genes)
        LOGGER.info(
            f"{label} | {describe_evaluation(individual)} | "
            f"evaluations {evaluator.evaluation_count}/{config.max_evaluations}"
        )
        individuals.append(individual)
    return individuals


def describe_generation_start(
    generation: int, candidate_count: int, elite_count: int, config: GeneticAlgorithmConfig
) -> str:
    heading = f"=== Generation {generation}/{config.generations}"
    if generation == 1:
        baseline_note = ", including the baseline" if config.include_baseline else ""
        return f"{heading}: initial population of {candidate_count} individuals{baseline_note}"
    return f"{heading}: {candidate_count} offspring + {elite_count} elite carried over"


def estimate_remaining_seconds(record: GenerationRecord, config: GeneticAlgorithmConfig) -> float:
    remaining_evaluations = (config.generations - record.generation) * config.offspring_per_generation
    seconds_per_evaluation = record.elapsed_seconds / max(record.evaluations_so_far, 1)
    return remaining_evaluations * seconds_per_evaluation


def describe_generation_summary(record: GenerationRecord, config: GeneticAlgorithmConfig) -> str:
    summary = (
        f"=== Generation {record.generation}/{config.generations} done | "
        f"best {record.best.fitness:.4f} | mean {record.mean_fitness:.4f} | "
        f"best so far {record.best_so_far.fitness:.4f} "
        f"({format_hyperparameters(record.best_so_far.hyperparameters)}) | "
        f"evaluations {record.evaluations_so_far}/{config.max_evaluations} | "
        f"elapsed {format_duration(record.elapsed_seconds)}"
    )
    if record.generation == config.generations:
        return summary
    return f"{summary} | ETA ~{format_duration(estimate_remaining_seconds(record, config))}"


def evolve(
    config: GeneticAlgorithmConfig, fitness_function: FitnessFunction
) -> Iterator[GenerationRecord]:
    rng = random.Random(config.seed)
    evaluator = CachedFitnessEvaluator(fitness_function)
    start_time = time.perf_counter()
    elites: List[Individual] = []
    candidate_genes = initial_population_genes(config, rng)
    best_so_far: Optional[Individual] = None

    for generation in range(1, config.generations + 1):
        LOGGER.info(describe_generation_start(generation, len(candidate_genes), len(elites), config))
        offspring = evaluate_candidates(candidate_genes, evaluator, generation, config)
        population = rank_by_fitness(elites + offspring)
        if best_so_far is None or population[0].fitness > best_so_far.fitness:
            best_so_far = population[0]
        record = GenerationRecord(
            generation=generation,
            population=population,
            best_so_far=best_so_far,
            evaluations_so_far=evaluator.evaluation_count,
            elapsed_seconds=time.perf_counter() - start_time,
        )
        LOGGER.info(describe_generation_summary(record, config))
        yield record
        elites = carry_over_elites(population, config.elite_size)
        candidate_genes = breed_offspring(
            population,
            config.offspring_per_generation,
            evaluator.known_hyperparameters,
            config,
            rng,
        )
