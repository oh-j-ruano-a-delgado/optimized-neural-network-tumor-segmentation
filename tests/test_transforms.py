"""Tests for the preprocessing and albumentations-based augmentation in ``src.dataset.transforms``.

The synthetic slice is a uint16 image (Figshare PNGs are 16-bit) with a bright elliptical "head"
on a black background and a rectangular tumor of class 2. Reproducibility is checked through
Torch's global generator because the augmentation pipeline reseeds itself from it on every call,
and the pickling test mimics how each DataLoader worker receives its own copy of the transform.
"""

import pickle
from dataclasses import replace
from typing import Tuple

import cv2
import numpy as np
import pytest
import torch

from src.dataset.transforms import (
    ADVANCED_AUGMENTATION,
    AUGMENTATION_PRESETS,
    BASIC_AUGMENTATION,
    SegmentationTransform,
    build_augmentation_pipeline,
    build_evaluation_transform,
    build_training_transform,
    resolve_augmentation_config,
)

IMAGE_SIZE = (64, 64)
TUMOR_CLASS = 2


def build_synthetic_slice() -> Tuple[np.ndarray, np.ndarray]:
    image = np.zeros((96, 112), dtype=np.uint16)
    cv2.ellipse(image, (56, 48), (44, 36), 0, 0, 360, 30000, thickness=-1)
    image[40:60, 50:70] = 52000
    mask = np.zeros((96, 112), dtype=np.uint8)
    mask[40:60, 50:70] = TUMOR_CLASS
    return image, mask


def apply_with_seed(
    transform: SegmentationTransform, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    image, mask = build_synthetic_slice()
    torch.manual_seed(seed)
    return transform(image, mask)


@pytest.mark.parametrize("preset", AUGMENTATION_PRESETS)
def test_every_preset_returns_expected_tensor_format(preset: str) -> None:
    transform = build_training_transform(IMAGE_SIZE, augmentation=preset)
    for seed in range(5):
        image, mask = apply_with_seed(transform, seed)
        assert image.shape == (1, *IMAGE_SIZE)
        assert image.dtype == torch.float32
        assert mask.shape == IMAGE_SIZE
        assert mask.dtype == torch.int64
        assert set(mask.unique().tolist()) <= {0, TUMOR_CLASS}
        assert torch.isfinite(image).all()
        assert float(image.min()) >= 0.0
        assert float(image.max()) <= 1.0


def test_none_preset_matches_evaluation_transform() -> None:
    training_image, training_mask = apply_with_seed(
        build_training_transform(IMAGE_SIZE, augmentation="none"), 0
    )
    evaluation_image, evaluation_mask = apply_with_seed(build_evaluation_transform(IMAGE_SIZE), 0)
    assert torch.equal(training_image, evaluation_image)
    assert torch.equal(training_mask, evaluation_mask)


def test_evaluation_transform_does_not_consume_torch_random_state() -> None:
    torch.manual_seed(3)
    expected_next_draw = torch.rand(1)
    torch.manual_seed(3)
    image, mask = build_synthetic_slice()
    build_evaluation_transform(IMAGE_SIZE)(image, mask)
    assert torch.equal(torch.rand(1), expected_next_draw)


@pytest.mark.parametrize("preset", ("basic", "advanced"))
def test_augmentation_is_reproducible_with_the_same_torch_seed(preset: str) -> None:
    transform = build_training_transform(IMAGE_SIZE, augmentation=preset)
    first_image, first_mask = apply_with_seed(transform, 11)
    second_image, second_mask = apply_with_seed(transform, 11)
    assert torch.equal(first_image, second_image)
    assert torch.equal(first_mask, second_mask)


@pytest.mark.parametrize("preset", ("basic", "advanced"))
def test_augmentation_changes_with_the_torch_seed(preset: str) -> None:
    transform = build_training_transform(IMAGE_SIZE, augmentation=preset)
    outputs = [apply_with_seed(transform, seed)[0] for seed in range(6)]
    assert any(not torch.equal(outputs[0], other) for other in outputs[1:])


def test_pickled_copies_follow_their_own_torch_seed() -> None:
    transform = build_training_transform(IMAGE_SIZE, augmentation="advanced")
    first_worker = pickle.loads(pickle.dumps(transform))
    second_worker = pickle.loads(pickle.dumps(transform))
    first_image, _ = apply_with_seed(first_worker, 101)
    second_image, _ = apply_with_seed(second_worker, 102)
    replayed_image, _ = apply_with_seed(second_worker, 101)
    assert not torch.equal(first_image, second_image)
    assert torch.equal(first_image, replayed_image)


def test_advanced_spatial_transforms_keep_mask_labels_valid() -> None:
    pipeline = build_augmentation_pipeline(ADVANCED_AUGMENTATION)
    image, mask = build_synthetic_slice()
    for seed in range(20):
        pipeline.spatial.set_random_seed(seed)
        augmented_image, augmented_mask = pipeline.apply_spatial(image, mask)
        assert augmented_image.dtype == np.uint16
        assert augmented_mask.dtype == np.uint8
        assert set(np.unique(augmented_mask).tolist()) <= {0, TUMOR_CLASS}


def test_intensity_transforms_leave_the_mask_untouched() -> None:
    transform = build_training_transform(IMAGE_SIZE, augmentation="advanced")
    without_intensity = SegmentationTransform(
        IMAGE_SIZE,
        augmentation=build_augmentation_pipeline(
            replace(
                ADVANCED_AUGMENTATION,
                brightness_contrast_probability=0.0,
                gamma_probability=0.0,
                noise_probability=0.0,
            )
        ),
    )
    for seed in range(5):
        _, mask_with_intensity = apply_with_seed(transform, seed)
        _, mask_without_intensity = apply_with_seed(without_intensity, seed)
        assert torch.equal(mask_with_intensity, mask_without_intensity)


def test_presets_select_the_documented_transforms() -> None:
    basic = build_augmentation_pipeline(BASIC_AUGMENTATION)
    advanced = build_augmentation_pipeline(ADVANCED_AUGMENTATION)
    assert [type(item).__name__ for item in basic.spatial.transforms] == [
        "HorizontalFlip",
        "VerticalFlip",
        "Affine",
    ]
    assert not basic.has_intensity_transforms
    assert [type(item).__name__ for item in advanced.spatial.transforms] == [
        "HorizontalFlip",
        "Affine",
        "OneOf",
    ]
    assert [type(item).__name__ for item in advanced.intensity.transforms] == [
        "RandomBrightnessContrast",
        "RandomGamma",
        "GaussNoise",
    ]


def test_intensity_augmentation_rejects_z_score_normalization() -> None:
    with pytest.raises(ValueError, match="minmax"):
        build_training_transform(IMAGE_SIZE, normalization="zscore", augmentation="advanced")


def test_basic_augmentation_allows_z_score_normalization() -> None:
    transform = build_training_transform(IMAGE_SIZE, normalization="zscore", augmentation="basic")
    image, _ = apply_with_seed(transform, 0)
    assert torch.isfinite(image).all()


def test_resolve_augmentation_config_accepts_configs_and_none() -> None:
    assert resolve_augmentation_config(None) is None
    assert resolve_augmentation_config("none") is None
    assert resolve_augmentation_config(ADVANCED_AUGMENTATION) is ADVANCED_AUGMENTATION


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown augmentation preset"):
        build_training_transform(IMAGE_SIZE, augmentation="extreme")
