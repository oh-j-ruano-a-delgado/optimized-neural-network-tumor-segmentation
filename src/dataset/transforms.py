"""Preprocessing and data augmentation that turn a raw MRI slice and its mask into tensors.

``SegmentationTransform`` maps a raw (image, mask) pair to an image tensor of shape (1, H, W)
float32 and a class-index mask of shape (H, W) int64 with values in [0, NUM_CLASSES). The steps
are, in order: grayscale conversion, background-margin crop, resize, spatial augmentation,
intensity normalization, intensity augmentation.

Preprocessing notes:
    Min-max normalization clips intensities to the 0.5th-99.5th percentile first, so a few hot
    pixels cannot compress the whole brain into a narrow intensity band after rescaling.
    The foreground detector applies a morphological opening to erase background noise and
    burned-in annotations that would otherwise stretch the crop far beyond the head, and it keeps
    every sizeable connected component (not only the largest) so a skull outline split into
    several blobs never loses a hemisphere.
    Masks are always resized and warped with nearest-neighbour interpolation so every pixel stays
    a valid class id instead of a blend of neighbouring labels.

Augmentation (``albumentations`` 2.0.8, the last MIT-licensed release) is built from two blocks:
    basic spatial     horizontal and vertical flips (p = 0.5 each) and a rotation of up to
                      +-15 degrees on every sample (the original Phase 1 policy).
    advanced spatial  horizontal flip (p = 0.5); affine rotation +-15 degrees, scale 0.9-1.1 and
                      translation +-6% (p = 0.7); one of ElasticTransform(alpha = 200,
                      sigma = 8) or GridDistortion(num_steps = 5, distort_limit = 0.15) (p = 0.3).
    intensity         applied to the normalized image only: RandomBrightnessContrast(+-0.15,
                      p = 0.5), RandomGamma(80-120, p = 0.3) and GaussNoise(std 0.01-0.03,
                      p = 0.2).
The presets combine them as a 2 x 2 factorial design (spatial block x intensity on/off):
    "none"                no augmentation.
    "basic"               basic spatial, no intensity.
    "advanced_spatial"    advanced spatial, no intensity.
    "advanced_intensity"  basic spatial + intensity.
    "advanced"            advanced spatial + intensity.
Each of the two isolation presets differs from "basic" or "advanced" in exactly one block, which
is what attributes an effect (such as the pituitary-to-meningioma confusion seen with
"advanced") to the spatial or to the intensity transforms. "advanced_intensity" keeps the basic
flips and rotation on purpose: an intensity-only preset would also remove every geometric
augmentation, so any change could not be attributed to the intensity transforms alone.
    The vertical flip is dropped from the advanced spatial block because Figshare contains
    sagittal and coronal slices, and flipping them upside down produces anatomy that never occurs
    at test time.
    The deformation strengths were measured on a 256 x 256 grid: the elastic setting displaces
    pixels by about 1.9 px on average (4.4 px at the 99th percentile) and the grid setting by about
    4.7 px (7.9 px), whereas the library default distort_limit of 0.3 moves them about 9 px, which
    is too strong for small pituitary tumors, and the default elastic setting moves nothing.

Ordering and randomness:
    Spatial transforms run before normalization, on the raw (uint8 or uint16) intensities, so the
    zero-valued border they introduce matches the real black background. Intensity transforms run
    after min-max normalization because they assume float images in [0, 1] and clip to that range;
    they would corrupt z-score images, so combining them with z-score normalization is rejected.
    Albumentations keeps its own random generator, and pickled copies of a pipeline (one per
    DataLoader worker) replay identical augmentations. Every call therefore reseeds the pipeline
    from Torch's global generator, which PyTorch reseeds per DataLoader worker and which
    ``set_global_seed`` resets, so workers draw different augmentations while seeded runs
    (including every metaheuristic fitness evaluation) stay reproducible.
    ``src/dataset/__init__.py`` sets ``NO_ALBUMENTATIONS_UPDATE`` before this module imports
    albumentations, so the library skips its online version check, which would otherwise issue a
    network request at import time in the main process and again in every spawned worker.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, List, Literal, Mapping, Optional, Tuple, Union, get_args

import albumentations as A
import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

NormalizationMode = Literal["minmax", "zscore"]
AugmentationPreset = Literal["none", "basic", "advanced_spatial", "advanced_intensity", "advanced"]
ImageSize = Tuple[int, int]
FloatRange = Tuple[float, float]

DEFAULT_IMAGE_SIZE: Final[ImageSize] = (256, 256)
LOWER_INTENSITY_PERCENTILE: Final[float] = 0.5
UPPER_INTENSITY_PERCENTILE: Final[float] = 99.5
NORMALIZATION_EPSILON: Final[float] = 1e-8
MINIMUM_CROP_SIDE: Final[int] = 32
AUGMENTATION_PRESETS: Final[Tuple[str, ...]] = get_args(AugmentationPreset)
DEFAULT_AUGMENTATION_PRESET: Final[AugmentationPreset] = "basic"
MAX_AUGMENTATION_SEED: Final[int] = 2**31 - 2
BORDER_FILL_VALUE: Final[int] = 0


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
    scale_range: FloatRange = (1.0, 1.0)
    max_translation_fraction: float = 0.0
    affine_probability: float = 1.0
    elastic_alpha: float = 200.0
    elastic_sigma: float = 8.0
    grid_distortion_steps: int = 5
    grid_distortion_limit: float = 0.15
    deformation_probability: float = 0.0
    brightness_limit: float = 0.15
    contrast_limit: float = 0.15
    brightness_contrast_probability: float = 0.0
    gamma_limit: FloatRange = (80.0, 120.0)
    gamma_probability: float = 0.0
    noise_std_range: FloatRange = (0.01, 0.03)
    noise_probability: float = 0.0


BASIC_AUGMENTATION: Final[AugmentationConfig] = AugmentationConfig()
ADVANCED_AUGMENTATION: Final[AugmentationConfig] = AugmentationConfig(
    vertical_flip_probability=0.0,
    scale_range=(0.9, 1.1),
    max_translation_fraction=0.06,
    affine_probability=0.7,
    deformation_probability=0.3,
    brightness_contrast_probability=0.5,
    gamma_probability=0.3,
    noise_probability=0.2,
)
ADVANCED_SPATIAL_AUGMENTATION: Final[AugmentationConfig] = replace(
    ADVANCED_AUGMENTATION,
    brightness_contrast_probability=0.0,
    gamma_probability=0.0,
    noise_probability=0.0,
)
ADVANCED_INTENSITY_AUGMENTATION: Final[AugmentationConfig] = replace(
    BASIC_AUGMENTATION,
    brightness_contrast_probability=ADVANCED_AUGMENTATION.brightness_contrast_probability,
    gamma_probability=ADVANCED_AUGMENTATION.gamma_probability,
    noise_probability=ADVANCED_AUGMENTATION.noise_probability,
)
AUGMENTATION_CONFIGS: Final[Mapping[str, Optional[AugmentationConfig]]] = MappingProxyType(
    {
        "none": None,
        "basic": BASIC_AUGMENTATION,
        "advanced_spatial": ADVANCED_SPATIAL_AUGMENTATION,
        "advanced_intensity": ADVANCED_INTENSITY_AUGMENTATION,
        "advanced": ADVANCED_AUGMENTATION,
    }
)


def build_spatial_transforms(config: AugmentationConfig) -> List[A.BasicTransform]:
    candidates: List[A.BasicTransform] = [
        A.HorizontalFlip(p=config.horizontal_flip_probability),
        A.VerticalFlip(p=config.vertical_flip_probability),
        A.Affine(
            rotate=(-config.max_rotation_degrees, config.max_rotation_degrees),
            scale=config.scale_range,
            translate_percent=(-config.max_translation_fraction, config.max_translation_fraction),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=BORDER_FILL_VALUE,
            fill_mask=BORDER_FILL_VALUE,
            p=config.affine_probability,
        ),
        A.OneOf(
            [
                A.ElasticTransform(
                    alpha=config.elastic_alpha,
                    sigma=config.elastic_sigma,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=BORDER_FILL_VALUE,
                    fill_mask=BORDER_FILL_VALUE,
                ),
                A.GridDistortion(
                    num_steps=config.grid_distortion_steps,
                    distort_limit=config.grid_distortion_limit,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=BORDER_FILL_VALUE,
                    fill_mask=BORDER_FILL_VALUE,
                ),
            ],
            p=config.deformation_probability,
        ),
    ]
    return [transform for transform in candidates if transform.p > 0]


def build_intensity_transforms(config: AugmentationConfig) -> List[A.BasicTransform]:
    candidates: List[A.BasicTransform] = [
        A.RandomBrightnessContrast(
            brightness_limit=config.brightness_limit,
            contrast_limit=config.contrast_limit,
            p=config.brightness_contrast_probability,
        ),
        A.RandomGamma(gamma_limit=config.gamma_limit, p=config.gamma_probability),
        A.GaussNoise(std_range=config.noise_std_range, p=config.noise_probability),
    ]
    return [transform for transform in candidates if transform.p > 0]


@dataclass(frozen=True)
class AugmentationPipeline:
    spatial: A.Compose
    intensity: A.Compose

    @property
    def has_intensity_transforms(self) -> bool:
        return len(self.intensity.transforms) > 0

    def reseed_from_torch(self) -> None:
        seed = int(torch.randint(0, MAX_AUGMENTATION_SEED, (1,)).item())
        self.spatial.set_random_seed(seed)
        self.intensity.set_random_seed(seed + 1)

    def apply_spatial(
        self, image: NDArray[np.generic], mask: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.generic], NDArray[np.uint8]]:
        augmented = self.spatial(image=np.ascontiguousarray(image), mask=np.ascontiguousarray(mask))
        return np.ascontiguousarray(augmented["image"]), np.ascontiguousarray(augmented["mask"])

    def apply_intensity(self, image: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self.has_intensity_transforms:
            return image
        augmented = self.intensity(image=np.ascontiguousarray(image))["image"]
        return np.ascontiguousarray(augmented, dtype=np.float32)


def build_augmentation_pipeline(config: AugmentationConfig) -> AugmentationPipeline:
    return AugmentationPipeline(
        spatial=A.Compose(build_spatial_transforms(config)),
        intensity=A.Compose(build_intensity_transforms(config)),
    )


def resolve_augmentation_config(
    augmentation: Union[AugmentationPreset, AugmentationConfig, None],
) -> Optional[AugmentationConfig]:
    if augmentation is None or isinstance(augmentation, AugmentationConfig):
        return augmentation
    if augmentation not in AUGMENTATION_CONFIGS:
        raise ValueError(
            f"Unknown augmentation preset '{augmentation}', expected one of {AUGMENTATION_PRESETS}."
        )
    return AUGMENTATION_CONFIGS[augmentation]


def to_grayscale(image: NDArray[np.generic]) -> NDArray[np.generic]:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    conversion_code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, conversion_code)


def rescale_min_max(image: NDArray[np.generic]) -> NDArray[np.float32]:
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
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

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
    resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask


def to_tensors(image: NDArray[np.float32], mask: NDArray[np.uint8]) -> Tuple[Tensor, Tensor]:
    image_tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0)
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
    return image_tensor, mask_tensor


@dataclass(frozen=True)
class SegmentationTransform:
    image_size: ImageSize = DEFAULT_IMAGE_SIZE
    normalization: NormalizationMode = "minmax"
    augmentation: Optional[AugmentationPipeline] = None
    crop_margins: bool = True

    def __post_init__(self) -> None:
        if (
            self.augmentation is not None
            and self.augmentation.has_intensity_transforms
            and self.normalization != "minmax"
        ):
            raise ValueError(
                "Intensity augmentations require 'minmax' normalization, "
                f"got '{self.normalization}'."
            )

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
        if self.augmentation is None:
            return to_tensors(normalize_intensity(grayscale_image, self.normalization), mask)

        self.augmentation.reseed_from_torch()
        grayscale_image, mask = self.augmentation.apply_spatial(grayscale_image, mask)
        normalized_image = normalize_intensity(grayscale_image, self.normalization)
        return to_tensors(self.augmentation.apply_intensity(normalized_image), mask)


def build_training_transform(
    image_size: ImageSize = DEFAULT_IMAGE_SIZE,
    normalization: NormalizationMode = "minmax",
    augmentation: Union[AugmentationPreset, AugmentationConfig, None] = DEFAULT_AUGMENTATION_PRESET,
) -> SegmentationTransform:
    config = resolve_augmentation_config(augmentation)
    pipeline = None if config is None else build_augmentation_pipeline(config)
    return SegmentationTransform(image_size, normalization, pipeline)


def build_evaluation_transform(
    image_size: ImageSize = DEFAULT_IMAGE_SIZE, normalization: NormalizationMode = "minmax"
) -> SegmentationTransform:
    return SegmentationTransform(image_size, normalization, augmentation=None)
