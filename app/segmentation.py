"""Model loading and segmentation helpers shared by the Streamlit interface and its tests.

The module keeps every piece of inference logic free of Streamlit so it can be unit
tested on CPU and reused by future front ends.
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Final, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.dataset.constants import BACKGROUND_LABEL, TUMOR_CLASS_NAMES
from src.dataset.dataloader import read_image_file
from src.dataset.transforms import (
    ImageSize,
    SegmentationTransform,
    build_evaluation_transform,
    to_grayscale,
)
from src.models.unet import UNet
from src.utils.checkpoint import (
    TrainingCheckpoint,
    build_model_from_checkpoint,
    load_training_checkpoint,
)
from src.utils.training import resolve_device

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODELS_DIR: Final[Path] = PROJECT_ROOT / "app" / "models"
SAMPLES_DIR: Final[Path] = PROJECT_ROOT / "app" / "samples"

BASELINE_MODEL_KEY: Final[str] = "baseline"
OPTIMIZED_MODEL_KEY: Final[str] = "optimized"
MODEL_CHECKPOINTS: Final[Mapping[str, Path]] = MappingProxyType(
    {
        BASELINE_MODEL_KEY: MODELS_DIR / "baseline_unet.pth",
        OPTIMIZED_MODEL_KEY: MODELS_DIR / "optimized_unet.pth",
    }
)
MODEL_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        BASELINE_MODEL_KEY: "Baseline (manual U-Net)",
        OPTIMIZED_MODEL_KEY: "Optimized (genetic algorithm)",
    }
)

BACKGROUND_COLOR: Final[Tuple[int, int, int]] = (0, 0, 0)
TUMOR_COLORS: Final[Mapping[int, Tuple[int, int, int]]] = MappingProxyType(
    {
        1: (255, 107, 53),
        2: (156, 100, 255),
        3: (0, 230, 118),
    }
)
DEFAULT_OVERLAY_ALPHA: Final[float] = 0.45
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.0
PERCENTAGE_SCALE: Final[float] = 100.0
CONFIDENCE_HIGH_THRESHOLD: Final[float] = 0.90
CONFIDENCE_MEDIUM_THRESHOLD: Final[float] = 0.75


@dataclass(frozen=True)
class SegmentationModel:
    key: str
    label: str
    checkpoint: TrainingCheckpoint
    network: UNet
    device: torch.device
    transform: SegmentationTransform

    @property
    def image_size(self) -> ImageSize:
        side = self.checkpoint.image_size
        return side, side


@dataclass(frozen=True)
class PredictionResult:
    mask: NDArray[np.uint8]
    probabilities: NDArray[np.float32]
    confidence: NDArray[np.float32]
    class_pixel_percentages: Mapping[str, float]


@dataclass(frozen=True)
class PreprocessedImage:
    tensor: Tensor
    display_image: NDArray[np.uint8]


@dataclass(frozen=True)
class SampleImage:
    class_name: str
    path: Path


def available_models(
    checkpoints: Mapping[str, Path] = MODEL_CHECKPOINTS,
) -> Dict[str, bool]:
    return {key: path.is_file() for key, path in checkpoints.items()}


def load_model(
    model_key: str,
    device: Optional[torch.device] = None,
    checkpoints: Mapping[str, Path] = MODEL_CHECKPOINTS,
) -> SegmentationModel:
    return load_model_from_checkpoint(model_key, checkpoints[model_key], device)


def load_model_from_checkpoint(
    model_key: str,
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> SegmentationModel:
    resolved_device = resolve_device("auto") if device is None else device
    checkpoint = load_training_checkpoint(checkpoint_path)
    network = build_model_from_checkpoint(checkpoint, resolved_device)
    side = checkpoint.image_size
    transform = build_evaluation_transform((side, side))
    return SegmentationModel(
        key=model_key,
        label=MODEL_LABELS.get(model_key, model_key),
        checkpoint=checkpoint,
        network=network,
        device=resolved_device,
        transform=transform,
    )


def read_sample_image(image_path: Path) -> NDArray[np.generic]:
    return to_grayscale(read_image_file(image_path, cv2.IMREAD_UNCHANGED))


def decode_image_bytes(image_bytes: bytes) -> NDArray[np.generic]:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("The uploaded file is not a decodable PNG image.")
    return to_grayscale(image)


def discover_sample_images(samples_dir: Path = SAMPLES_DIR) -> List[SampleImage]:
    return [
        SampleImage(class_name=class_name, path=image_path)
        for class_name in TUMOR_CLASS_NAMES.values()
        for image_path in sorted((samples_dir / class_name).glob("*.png"))
    ]


def preprocess_image(image: NDArray[np.generic], model: SegmentationModel) -> PreprocessedImage:
    grayscale = to_grayscale(image)
    # The evaluation transform only reads the mask to widen the crop around the
    # tumor; at inference time the head silhouette alone defines that crop.
    empty_mask = np.zeros(grayscale.shape, dtype=np.uint8)
    image_tensor, _ = model.transform(grayscale, empty_mask)
    display_image = np.clip(image_tensor[0].numpy() * 255.0, 0.0, 255.0).astype(np.uint8)
    return PreprocessedImage(
        tensor=image_tensor.unsqueeze(0).to(model.device),
        display_image=display_image,
    )


def predict(model: SegmentationModel, image_tensor: Tensor) -> PredictionResult:
    with torch.no_grad():
        logits = model.network(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].float().cpu().numpy()
    mask = probabilities.argmax(axis=0).astype(np.uint8)
    confidence = probabilities.max(axis=0).astype(np.float32)
    return PredictionResult(
        mask=mask,
        probabilities=probabilities,
        confidence=confidence,
        class_pixel_percentages=class_pixel_percentages(mask),
    )


def class_pixel_percentages(mask: NDArray[np.uint8]) -> Dict[str, float]:
    total_pixels = float(mask.size)
    percentages = {
        "background": PERCENTAGE_SCALE
        * float(np.count_nonzero(mask == BACKGROUND_LABEL))
        / total_pixels
    }
    for label, class_name in TUMOR_CLASS_NAMES.items():
        percentages[class_name] = (
            PERCENTAGE_SCALE * float(np.count_nonzero(mask == label)) / total_pixels
        )
    return percentages


def dominant_class_name(mask: NDArray[np.uint8]) -> Optional[str]:
    counts = [
        (np.count_nonzero(mask == label), class_name)
        for label, class_name in TUMOR_CLASS_NAMES.items()
    ]
    tumor_count, class_name = max(counts)
    return class_name if tumor_count > 0 else None


def average_tumor_confidence(
    mask: NDArray[np.uint8], confidence: NDArray[np.float32]
) -> Optional[float]:
    tumor_pixels = mask != BACKGROUND_LABEL
    if not np.any(tumor_pixels):
        return None
    return float(confidence[tumor_pixels].mean())


def classify_confidence(average_confidence: float) -> str:
    if average_confidence >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if average_confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def apply_confidence_threshold(
    mask: NDArray[np.uint8],
    confidence: NDArray[np.float32],
    threshold: float,
) -> NDArray[np.uint8]:
    if threshold <= 0.0:
        return mask
    filtered_mask = mask.copy()
    filtered_mask[confidence < threshold] = BACKGROUND_LABEL
    return filtered_mask


def to_display_image(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv2.cvtColor(to_grayscale(image), cv2.COLOR_GRAY2RGB)


def build_mask_image(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask == BACKGROUND_LABEL] = BACKGROUND_COLOR
    for label, color in TUMOR_COLORS.items():
        image[mask == label] = color
    return image


def build_overlay(
    display_image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> NDArray[np.uint8]:
    rgb_image = to_display_image(display_image) if display_image.ndim == 2 else display_image
    overlay = rgb_image.astype(np.float32)
    tumor_pixels = mask != BACKGROUND_LABEL
    color_layer = build_mask_image(mask).astype(np.float32)
    overlay[tumor_pixels] = (
        (1.0 - alpha) * overlay[tumor_pixels] + alpha * color_layer[tumor_pixels]
    )
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def build_confidence_image(confidence: NDArray[np.float32]) -> NDArray[np.uint8]:
    scaled_confidence = np.clip(confidence, 0.0, 1.0) * 255.0
    heatmap = cv2.applyColorMap(scaled_confidence.astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def encode_png(image_rgb: NDArray[np.uint8]) -> bytes:
    success, buffer = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Could not encode the image as PNG.")
    return buffer.tobytes()
