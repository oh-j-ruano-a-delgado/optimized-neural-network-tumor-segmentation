"""Evaluate U-Net hyperparameters using a fast early-convergence training loop.

Usage (imported by genetic algorithm):
    from src.optimization.fitness import evaluate_fitness
"""

import gc
import math
import time
from dataclasses import dataclass
from typing import Final, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from ..dataset.constants import NUM_CLASSES
from ..dataset.dataloader import SegmentationBatch, SegmentationDataLoaders, get_dataloaders
from ..dataset.transforms import DEFAULT_IMAGE_SIZE, ImageSize
from ..models.losses import (
    DEFAULT_DICE_WEIGHT,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_FOCAL_WEIGHT,
    DiceFocalLoss,
)
from ..models.unet import UNet
from ..utils.metrics import SegmentationMetrics
from ..utils.training import (
    build_grad_scaler,
    evaluate_epoch,
    resolve_amp_mode,
    resolve_device,
    resolve_max_batches,
    set_global_seed,
    train_epoch,
)

TrialStatus = Literal["completed", "diverged", "out_of_memory"]

UNIT_INTERVAL: Final[Tuple[float, float]] = (0.0, 1.0)
MIN_FITNESS_EPOCHS: Final[int] = 1
MAX_FITNESS_EPOCHS: Final[int] = 5
DEFAULT_FITNESS_EPOCHS: Final[int] = 3
DEFAULT_BATCH_SIZE: Final[int] = 8
DEFAULT_VALIDATION_FRACTION: Final[float] = 0.2
DEFAULT_SEED: Final[int] = 42
INFEASIBLE_FITNESS: Final[float] = 0.0


@dataclass(frozen=True)
class LogUniformGene:
    name: str
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        if not 0.0 < self.lower_bound < self.upper_bound:
            raise ValueError(
                f"Gene '{self.name}' needs 0 < lower_bound < upper_bound, "
                f"got [{self.lower_bound}, {self.upper_bound}]."
            )

    @property
    def log_bounds(self) -> Tuple[float, float]:
        return math.log10(self.lower_bound), math.log10(self.upper_bound)

    def decode(self, unit_value: float) -> float:
        log_lower, log_upper = self.log_bounds
        return 10.0 ** (log_lower + unit_value * (log_upper - log_lower))

    def encode(self, value: float) -> float:
        if not self.lower_bound <= value <= self.upper_bound:
            raise ValueError(
                f"Gene '{self.name}' value {value} is outside "
                f"[{self.lower_bound}, {self.upper_bound}]."
            )
        log_lower, log_upper = self.log_bounds
        return (math.log10(value) - log_lower) / (log_upper - log_lower)


@dataclass(frozen=True)
class CategoricalGene:
    name: str
    choices: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError(f"Gene '{self.name}' needs at least one choice.")

    def decode(self, unit_value: float) -> int:
        choice_index = min(int(unit_value * len(self.choices)), len(self.choices) - 1)
        return self.choices[choice_index]

    def encode(self, value: int) -> float:
        if value not in self.choices:
            raise ValueError(f"Gene '{self.name}' value {value} is not one of {self.choices}.")
        choice_index = self.choices.index(value)
        bin_center = (choice_index + 0.5) / len(self.choices)
        return bin_center


Gene = Union[LogUniformGene, CategoricalGene]

LEARNING_RATE_GENE: Final[LogUniformGene] = LogUniformGene("learning_rate", 1e-4, 1e-2)
WEIGHT_DECAY_GENE: Final[LogUniformGene] = LogUniformGene("weight_decay", 1e-6, 1e-3)
BASE_FILTERS_GENE: Final[CategoricalGene] = CategoricalGene("base_filters", (8, 16, 32))
DEPTH_GENE: Final[CategoricalGene] = CategoricalGene("depth", (3, 4, 5))
SEARCH_SPACE: Final[Tuple[Gene, ...]] = (
    LEARNING_RATE_GENE,
    WEIGHT_DECAY_GENE,
    BASE_FILTERS_GENE,
    DEPTH_GENE,
)


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float
    weight_decay: float
    base_filters: int
    depth: int


BASELINE_HYPERPARAMETERS: Final[Hyperparameters] = Hyperparameters(
    learning_rate=1e-3, weight_decay=1e-4, base_filters=16, depth=4
)


@dataclass(frozen=True)
class FitnessSettings:
    epochs: int = DEFAULT_FITNESS_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    image_size: ImageSize = DEFAULT_IMAGE_SIZE
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    dice_weight: float = DEFAULT_DICE_WEIGHT
    focal_weight: float = DEFAULT_FOCAL_WEIGHT
    focal_gamma: float = DEFAULT_FOCAL_GAMMA
    max_train_batches: Optional[int] = None
    max_validation_batches: Optional[int] = None
    num_workers: int = 0
    seed: int = DEFAULT_SEED
    device: str = "auto"
    amp: bool = False

    def __post_init__(self) -> None:
        if not MIN_FITNESS_EPOCHS <= self.epochs <= MAX_FITNESS_EPOCHS:
            raise ValueError(
                f"epochs must be between {MIN_FITNESS_EPOCHS} and {MAX_FITNESS_EPOCHS} "
                f"for an early-convergence test, got {self.epochs}."
            )
        for batch_limit in (self.max_train_batches, self.max_validation_batches):
            if batch_limit is not None and batch_limit < 1:
                raise ValueError(f"Batch limits must be positive, got {batch_limit}.")
        deepest_downsampling_factor = 2 ** max(DEPTH_GENE.choices)
        if any(side % deepest_downsampling_factor for side in self.image_size):
            raise ValueError(
                f"image_size {self.image_size} must be divisible by "
                f"{deepest_downsampling_factor} to fit every depth in {DEPTH_GENE.choices}."
            )


DEFAULT_FITNESS_SETTINGS: Final[FitnessSettings] = FitnessSettings()


@dataclass(frozen=True)
class FitnessTrial:
    hyperparameters: Hyperparameters
    status: TrialStatus
    validation_dice_by_epoch: Tuple[float, ...]
    elapsed_seconds: float

    @property
    def fitness(self) -> float:
        if self.status != "completed" or not self.validation_dice_by_epoch:
            return INFEASIBLE_FITNESS
        return max(self.validation_dice_by_epoch)


def clip_to_unit_interval(gene_value: float) -> float:
    if not math.isfinite(gene_value):
        raise ValueError(f"Gene values must be finite, got {gene_value}.")
    lower_limit, upper_limit = UNIT_INTERVAL
    return min(max(float(gene_value), lower_limit), upper_limit)


def decode_genes(genes: Sequence[float]) -> Hyperparameters:
    if len(genes) != len(SEARCH_SPACE):
        gene_names = ", ".join(gene.name for gene in SEARCH_SPACE)
        raise ValueError(f"Expected {len(SEARCH_SPACE)} genes ({gene_names}), got {len(genes)}.")
    learning_rate, weight_decay, base_filters, depth = (
        clip_to_unit_interval(gene_value) for gene_value in genes
    )
    return Hyperparameters(
        learning_rate=LEARNING_RATE_GENE.decode(learning_rate),
        weight_decay=WEIGHT_DECAY_GENE.decode(weight_decay),
        base_filters=BASE_FILTERS_GENE.decode(base_filters),
        depth=DEPTH_GENE.decode(depth),
    )


def encode_hyperparameters(hyperparameters: Hyperparameters) -> List[float]:
    return [
        LEARNING_RATE_GENE.encode(hyperparameters.learning_rate),
        WEIGHT_DECAY_GENE.encode(hyperparameters.weight_decay),
        BASE_FILTERS_GENE.encode(hyperparameters.base_filters),
        DEPTH_GENE.encode(hyperparameters.depth),
    ]


def build_fitness_dataloaders(settings: FitnessSettings) -> SegmentationDataLoaders:
    return get_dataloaders(
        batch_size=settings.batch_size,
        image_size=settings.image_size,
        validation_fraction=settings.validation_fraction,
        num_workers=settings.num_workers,
        seed=settings.seed,
    )


def restart_shuffle_order(loader: DataLoader[SegmentationBatch], seed: int) -> None:
    if loader.generator is not None:
        loader.generator.manual_seed(seed)


def release_accelerator_memory(device: torch.device) -> None:
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()


def finite_or_infeasible(score: float) -> float:
    return score if math.isfinite(score) else INFEASIBLE_FITNESS


def train_and_validate(
    hyperparameters: Hyperparameters,
    dataloaders: SegmentationDataLoaders,
    settings: FitnessSettings,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[TrialStatus, Tuple[float, ...]]:
    model = UNet(
        num_classes=NUM_CLASSES,
        base_filters=hyperparameters.base_filters,
        depth=hyperparameters.depth,
    ).to(device)
    criterion = DiceFocalLoss(
        dice_weight=settings.dice_weight,
        focal_weight=settings.focal_weight,
        num_classes=NUM_CLASSES,
        gamma=settings.focal_gamma,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    scaler = build_grad_scaler(amp_enabled)
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)
    max_train_batches = resolve_max_batches(dataloaders.train, settings.max_train_batches)
    max_validation_batches = resolve_max_batches(
        dataloaders.validation, settings.max_validation_batches
    )

    validation_dice_by_epoch: List[float] = []
    for _ in range(settings.epochs):
        train_loss = train_epoch(
            model,
            dataloaders.train,
            criterion,
            optimizer,
            scaler,
            amp_enabled,
            device,
            max_train_batches,
        )
        if not math.isfinite(train_loss):
            return "diverged", tuple(validation_dice_by_epoch)
        _, report = evaluate_epoch(
            model,
            dataloaders.validation,
            criterion,
            metrics,
            amp_enabled,
            device,
            max_validation_batches,
        )
        validation_dice_by_epoch.append(finite_or_infeasible(report.mean_dice))
    return "completed", tuple(validation_dice_by_epoch)


def run_fitness_trial(
    hyperparameters: Hyperparameters,
    dataloaders: SegmentationDataLoaders,
    settings: FitnessSettings = DEFAULT_FITNESS_SETTINGS,
) -> FitnessTrial:
    set_global_seed(settings.seed)
    restart_shuffle_order(dataloaders.train, settings.seed)
    device = resolve_device(settings.device)
    amp_enabled = resolve_amp_mode(settings.amp, device)

    start_time = time.perf_counter()
    status: TrialStatus
    validation_dice_by_epoch: Tuple[float, ...]
    try:
        status, validation_dice_by_epoch = train_and_validate(
            hyperparameters, dataloaders, settings, device, amp_enabled
        )
    except torch.OutOfMemoryError:
        status, validation_dice_by_epoch = "out_of_memory", ()
    elapsed_seconds = time.perf_counter() - start_time
    release_accelerator_memory(device)

    return FitnessTrial(
        hyperparameters=hyperparameters,
        status=status,
        validation_dice_by_epoch=validation_dice_by_epoch,
        elapsed_seconds=elapsed_seconds,
    )


def evaluate_fitness(
    genes: Sequence[float],
    settings: FitnessSettings = DEFAULT_FITNESS_SETTINGS,
    dataloaders: Optional[SegmentationDataLoaders] = None,
) -> float:
    trial_dataloaders = build_fitness_dataloaders(settings) if dataloaders is None else dataloaders
    return run_fitness_trial(decode_genes(genes), trial_dataloaders, settings).fitness
