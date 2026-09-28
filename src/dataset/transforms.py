from dataclasses import dataclass
from typing import Final, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

NormalizationMode = Literal["minmax", "zscore"]
ImageSize = Tuple[int, int]

DEFAULT_IMAGE_SIZE: Final[ImageSize] = (256, 256)
LOWER_INTENSITY_PERCENTILE: Final[float] = 0.5
UPPER_INTENSITY_PERCENTILE: Final[float] = 99.5
NORMALIZATION_EPSILON: Final[float] = 1e-8
MINIMUM_CROP_SIDE: Final[int] = 32


@dataclass(frozen=True)
class BoundingBox:
    top: int
    bottom: int
    left: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            top=min(self.top, other.top),
            bottom=max(self.bottom, other.bottom),
            left=min(self.left, other.left),
            right=max(self.right, other.right),
        )

    def padded(self, padding: int, image_shape: Tuple[int, int]) -> "BoundingBox":
        image_height, image_width = image_shape
        return BoundingBox(
            top=max(self.top - padding, 0),
            bottom=min(self.bottom + padding, image_height),
            left=max(self.left - padding, 0),
            right=min(self.right + padding, image_width),
        )

    def crop(self, array: NDArray[np.generic]) -> NDArray[np.generic]:
        return array[self.top : self.bottom, self.left : self.right]


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    max_rotation_degrees: float = 15.0


DEFAULT_AUGMENTATION: Final[AugmentationConfig] = AugmentationConfig()


def to_grayscale(image: NDArray[np.generic]) -> NDArray[np.generic]:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    conversion_code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, conversion_code)


def rescale_min_max(image: NDArray[np.generic]) -> NDArray[np.float32]:
    # Percentile clipping keeps a few hot pixels from compressing the whole
    # brain into a narrow intensity band after rescaling.
    image_float = image.astype(np.float32)
    lower_bound, upper_bound = np.percentile(
        image_float, (LOWER_INTENSITY_PERCENTILE, UPPER_INTENSITY_PERCENTILE)
    )
    clipped = np.clip(image_float, lower_bound, upper_bound)
    return ((clipped - lower_bound) / (upper_bound - lower_bound + NORMALIZATION_EPSILON)).astype(
        np.float32
    )


def standardize_z_score(image: NDArray[np.generic]) -> NDArray[np.float32]:
    image_float = image.astype(np.float32)
    return (
        (image_float - image_float.mean()) / (image_float.std() + NORMALIZATION_EPSILON)
    ).astype(np.float32)


def normalize_intensity(
    image: NDArray[np.generic], mode: NormalizationMode = "minmax"
) -> NDArray[np.float32]:
    return standardize_z_score(image) if mode == "zscore" else rescale_min_max(image)


def compute_mask_bounding_box(mask: NDArray[np.generic]) -> Optional[BoundingBox]:
    foreground_rows, foreground_columns = np.nonzero(mask)
    if foreground_rows.size == 0:
        return None
    return BoundingBox(
        top=int(foreground_rows.min()),
        bottom=int(foreground_rows.max()) + 1,
        left=int(foreground_columns.min()),
        right=int(foreground_columns.max()) + 1,
    )


def compute_foreground_bounding_box(
    image: NDArray[np.generic],
    intensity_threshold: float = 0.08,
    minimum_component_ratio: float = 0.05,
    kernel_size: int = 5,
) -> Optional[BoundingBox]:
    image_uint8 = (rescale_min_max(to_grayscale(image)) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(image_uint8, (kernel_size, kernel_size), 0)
    _, foreground = cv2.threshold(blurred, intensity_threshold * 255, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Opening erases background noise and burned-in annotations that would
    # otherwise stretch the crop far beyond the head.
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keeping every sizeable component (not only the largest) avoids cutting away
    # a hemisphere when the skull outline is split into several blobs.
    largest_area = max(cv2.contourArea(contour) for contour in contours)
    significant_points = np.concatenate(
        [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= minimum_component_ratio * largest_area
        ]
    )
    left, top, width, height = cv2.boundingRect(significant_points)
    return BoundingBox(top=top, bottom=top + height, left=left, right=left + width)


def crop_background_margins(
    image: NDArray[np.generic],
    mask: NDArray[np.uint8],
    intensity_threshold: float = 0.08,
    padding: int = 4,
) -> Tuple[NDArray[np.generic], NDArray[np.uint8]]:
    foreground_box = compute_foreground_bounding_box(image, intensity_threshold)
    if foreground_box is None:
        return image, mask

    tumor_box = compute_mask_bounding_box(mask)
    crop_box = foreground_box if tumor_box is None else foreground_box.union(tumor_box)
    crop_box = crop_box.padded(padding, (mask.shape[0], mask.shape[1]))

    if crop_box.height < MINIMUM_CROP_SIDE or crop_box.width < MINIMUM_CROP_SIDE:
        return image, mask
    return crop_box.crop(image), crop_box.crop(mask)


def resize_image_and_mask(
    image: NDArray[np.generic], mask: NDArray[np.uint8], size: ImageSize = DEFAULT_IMAGE_SIZE
) -> Tuple[NDArray[np.generic], NDArray[np.uint8]]:
    target_height, target_width = size
    is_downscaling = image.shape[0] > target_height and image.shape[1] > target_width
    image_interpolation = cv2.INTER_AREA if is_downscaling else cv2.INTER_LINEAR
    resized_image = cv2.resize(
        image, (target_width, target_height), interpolation=image_interpolation
    )
    # Nearest-neighbour keeps every mask pixel a valid class id instead of
    # blending neighbouring labels into non-existent classes.
    resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask


def rotate_image_and_mask(
    image: NDArray[np.generic], mask: NDArray[np.uint8], angle_degrees: float
) -> Tuple[NDArray[np.generic], NDArray[np.uint8]]:
    height, width = mask.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_degrees, 1.0)
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_mask = cv2.warpAffine(
        mask,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated_image, rotated_mask


def sample_uniform(low: float, high: float) -> float:
    # Torch's RNG is reseeded per DataLoader worker whereas NumPy's is not, so
    # drawing from NumPy would make every worker emit identical augmentations.
    return float(torch.empty(1).uniform_(low, high).item())


def augment_image_and_mask(
    image: NDArray[np.generic], mask: NDArray[np.uint8], config: AugmentationConfig
) -> Tuple[NDArray[np.generic], NDArray[np.uint8]]:
    if sample_uniform(0.0, 1.0) < config.horizontal_flip_probability:
        image, mask = np.fliplr(image), np.fliplr(mask)
    if sample_uniform(0.0, 1.0) < config.vertical_flip_probability:
        image, mask = np.flipud(image), np.flipud(mask)
    if config.max_rotation_degrees > 0:
        angle = sample_uniform(-config.max_rotation_degrees, config.max_rotation_degrees)
        image, mask = rotate_image_and_mask(
            np.ascontiguousarray(image), np.ascontiguousarray(mask), angle
        )
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def to_tensors(image: NDArray[np.float32], mask: NDArray[np.uint8]) -> Tuple[Tensor, Tensor]:
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
    return image_tensor, mask_tensor


@dataclass(frozen=True)
class SegmentationTransform:
    """Maps a raw (image, mask) pair to an image tensor of shape (1, H, W) float32
    and a class-index mask of shape (H, W) int64 with values in [0, NUM_CLASSES)."""

    image_size: ImageSize = DEFAULT_IMAGE_SIZE
    normalization: NormalizationMode = "minmax"
    augmentation: Optional[AugmentationConfig] = None
    crop_margins: bool = True

    def __call__(
        self, image: NDArray[np.generic], mask: NDArray[np.uint8]
    ) -> Tuple[Tensor, Tensor]:
        grayscale_image = to_grayscale(image)
        if grayscale_image.shape != mask.shape:
            raise ValueError(
                f"Image shape {grayscale_image.shape} does not match mask shape {mask.shape}."
            )

        if self.crop_margins:
            grayscale_image, mask = crop_background_margins(grayscale_image, mask)
        grayscale_image, mask = resize_image_and_mask(grayscale_image, mask, self.image_size)
        # Augmenting before normalization keeps the zero-valued rotation border
        # equal to the real background in both min-max and z-score modes.
        if self.augmentation is not None:
            grayscale_image, mask = augment_image_and_mask(grayscale_image, mask, self.augmentation)
        return to_tensors(normalize_intensity(grayscale_image, self.normalization), mask)


def build_training_transform(
    image_size: ImageSize = DEFAULT_IMAGE_SIZE,
    normalization: NormalizationMode = "minmax",
    augmentation: AugmentationConfig = DEFAULT_AUGMENTATION,
) -> SegmentationTransform:
    return SegmentationTransform(image_size, normalization, augmentation)


def build_evaluation_transform(
    image_size: ImageSize = DEFAULT_IMAGE_SIZE, normalization: NormalizationMode = "minmax"
) -> SegmentationTransform:
    return SegmentationTransform(image_size, normalization, augmentation=None)
