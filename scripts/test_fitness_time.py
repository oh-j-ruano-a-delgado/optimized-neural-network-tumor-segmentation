"""Measure the execution time of one fitness evaluation with the baseline hyperparameters.

Usage (from the project root):
    python -m scripts.test_fitness_time [--epochs 3] [--device auto|cpu|cuda] [--amp] [--num-workers 0] [--limit-batches N]
"""

import argparse
import time
from typing import Final

from src.optimization.fitness import (
    BASELINE_HYPERPARAMETERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FITNESS_EPOCHS,
    FitnessSettings,
    Hyperparameters,
    build_fitness_dataloaders,
    encode_hyperparameters,
    evaluate_fitness,
)
from src.utils.training import resolve_amp_mode, resolve_device

SECONDS_PER_MINUTE: Final[int] = 60
SECONDS_PER_HOUR: Final[int] = 3600


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=DEFAULT_FITNESS_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def format_duration(total_seconds: float) -> str:
    hours, remaining_seconds = divmod(total_seconds, SECONDS_PER_HOUR)
    minutes, seconds = divmod(remaining_seconds, SECONDS_PER_MINUTE)
    return f"{total_seconds:.3f} s ({int(hours)}h {int(minutes):02d}m {seconds:06.3f}s)"


def describe_hyperparameters(hyperparameters: Hyperparameters) -> str:
    return (
        f"learning_rate={hyperparameters.learning_rate:.0e}, "
        f"weight_decay={hyperparameters.weight_decay:.0e}, "
        f"base_filters={hyperparameters.base_filters}, depth={hyperparameters.depth}"
    )


def main() -> None:
    arguments = parse_arguments()
    settings = FitnessSettings(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        max_train_batches=arguments.limit_batches,
        max_validation_batches=arguments.limit_batches,
        num_workers=arguments.num_workers,
        device=arguments.device,
        amp=arguments.amp,
    )
    device = resolve_device(settings.device)
    amp_enabled = resolve_amp_mode(settings.amp, device)

    loading_start = time.perf_counter()
    dataloaders = build_fitness_dataloaders(settings)
    loading_seconds = time.perf_counter() - loading_start
    print(
        f"Device: {device} | AMP: {'on' if amp_enabled else 'off'} | "
        f"workers: {settings.num_workers} | batch limit: {settings.max_train_batches or 'none'} | "
        f"train {len(dataloaders.train.dataset)} samples | "
        f"validation {len(dataloaders.validation.dataset)} samples"
    )
    print(
        f"Evaluating baseline ({describe_hyperparameters(BASELINE_HYPERPARAMETERS)}) "
        f"for {settings.epochs} epochs..."
    )

    evaluation_start = time.perf_counter()
    fitness = evaluate_fitness(
        encode_hyperparameters(BASELINE_HYPERPARAMETERS), settings, dataloaders
    )
    evaluation_seconds = time.perf_counter() - evaluation_start

    print(f"Fitness (best validation Dice): {fitness:.4f}")
    print(f"Dataset loading:    {format_duration(loading_seconds)}")
    print(f"Fitness evaluation: {format_duration(evaluation_seconds)}")
    print(f"Average per epoch:  {format_duration(evaluation_seconds / settings.epochs)}")
    print(f"Total:              {format_duration(loading_seconds + evaluation_seconds)}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(f"Error: {error}")
