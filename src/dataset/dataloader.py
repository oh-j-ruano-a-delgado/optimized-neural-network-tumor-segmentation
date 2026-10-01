from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Sequence, Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .constants import (
    IMAGES_DIRNAME,
    MASKS_DIRNAME,
    RAW_DATA_DIR,
    TESTING_SPLIT,
    TRAINING_SPLIT,
    TUMOR_CLASS_NAMES,
    parse_patient_id,
)
from .splits import split_patients_stratified
from .transforms import (
    DEFAULT_IMAGE_SIZE,
    ImageSize,
    NormalizationMode,
    SegmentationTransform,
    build_evaluation_transform,
    build_training_transform,
)

SegmentationBatch = Tuple[Tensor, Tensor]


@dataclass(frozen=True)
class SegmentationSample:
    image_path: Path
    mask_path: Path
    class_label: int
    patient_id: str


class SegmentationDataLoaders(NamedTuple):
    train: DataLoader[SegmentationBatch]
    validation: DataLoader[SegmentationBatch]
    test: DataLoader[SegmentationBatch]


def read_image_file(path: Path, read_flag: int) -> NDArray[np.generic]:
    # cv2.imread returns None for non-ASCII paths on Windows (e.g. "3º"), so the
    # bytes are loaded through NumPy and decoded in memory instead.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), read_flag)
    if image is None:
        raise OSError(f"Could not decode image file '{path}'.")
    return image


def discover_samples(split_dir: Path) -> List[SegmentationSample]:
    samples = [
        SegmentationSample(
            image_path=image_path,
            mask_path=split_dir / class_name / MASKS_DIRNAME / image_path.name,
            class_label=class_label,
            patient_id=parse_patient_id(image_path),
        )
        for class_label, class_name in TUMOR_CLASS_NAMES.items()
        for image_path in sorted((split_dir / class_name / IMAGES_DIRNAME).glob("*.png"))
    ]
    if not samples:
        raise FileNotFoundError(
            f"No samples found in '{split_dir}'. Run 'python -m src.dataset.download_data' first."
        )

    missing_masks = [sample.mask_path for sample in samples if not sample.mask_path.is_file()]
    if missing_masks:
        raise FileNotFoundError(
            f"{len(missing_masks)} images have no mask, e.g. '{missing_masks[0]}'."
        )
    return samples


def split_samples_by_patient(
    samples: Sequence[SegmentationSample], validation_fraction: float, seed: int
) -> Tuple[List[SegmentationSample], List[SegmentationSample]]:
    patient_labels = {sample.patient_id: sample.class_label for sample in samples}
    training_patients, _ = split_patients_stratified(patient_labels, validation_fraction, seed)
    training_samples = [sample for sample in samples if sample.patient_id in training_patients]
    validation_samples = [
        sample for sample in samples if sample.patient_id not in training_patients
    ]
    return training_samples, validation_samples


class BrainTumorSegmentationDataset(Dataset[SegmentationBatch]):
    def __init__(
        self, samples: Sequence[SegmentationSample], transform: SegmentationTransform
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SegmentationBatch:
        sample = self.samples[index]
        image = read_image_file(sample.image_path, cv2.IMREAD_UNCHANGED)
        mask = read_image_file(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        return self.transform(image, mask)


def build_dataloader(
    dataset: BrainTumorSegmentationDataset,
    batch_size: int,
    is_training: bool,
    num_workers: int,
    seed: int,
) -> DataLoader[SegmentationBatch]:
    # A seeded generator fixes the shuffle order and each worker's base seed, so
    # every metaheuristic fitness evaluation sees the same batches and augmentations.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        drop_last=is_training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def get_dataloaders(
    batch_size: int,
    image_size: ImageSize = DEFAULT_IMAGE_SIZE,
    validation_fraction: float = 0.2,
    normalization: NormalizationMode = "minmax",
    num_workers: int = 0,
    seed: int = 42,
    data_dir: Path = RAW_DATA_DIR,
) -> SegmentationDataLoaders:
    training_samples, validation_samples = split_samples_by_patient(
        discover_samples(data_dir / TRAINING_SPLIT), validation_fraction, seed
    )
    testing_samples = discover_samples(data_dir / TESTING_SPLIT)

    training_transform = build_training_transform(image_size, normalization)
    evaluation_transform = build_evaluation_transform(image_size, normalization)

    return SegmentationDataLoaders(
        train=build_dataloader(
            BrainTumorSegmentationDataset(training_samples, training_transform),
            batch_size,
            is_training=True,
            num_workers=num_workers,
            seed=seed,
        ),
        validation=build_dataloader(
            BrainTumorSegmentationDataset(validation_samples, evaluation_transform),
            batch_size,
            is_training=False,
            num_workers=num_workers,
            seed=seed,
        ),
        test=build_dataloader(
            BrainTumorSegmentationDataset(testing_samples, evaluation_transform),
            batch_size,
            is_training=False,
            num_workers=num_workers,
            seed=seed,
        ),
    )
