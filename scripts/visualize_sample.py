"""Plot a random MRI slice next to its mask and overlay to verify the prepared dataset.

Usage (from the project root):
    python -m scripts.visualize_sample [--split Training|Testing] [--seed 7] [--output sample.png]
"""

import argparse
import random
from pathlib import Path
from typing import Final, Mapping, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray

from src.dataset.constants import (
    BACKGROUND_LABEL,
    NUM_CLASSES,
    RAW_DATA_DIR,
    TESTING_SPLIT,
    TRAINING_SPLIT,
    TUMOR_CLASS_NAMES,
)
from src.dataset.dataloader import SegmentationSample, discover_samples, read_image_file
from src.dataset.transforms import rescale_min_max

BACKGROUND_COLOR: Final[str] = "#000000"
CLASS_COLORS: Final[Mapping[int, str]] = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
OVERLAY_ALPHA: Final[float] = 0.45


def build_class_colormap() -> Tuple[ListedColormap, BoundaryNorm]:
    colors = [BACKGROUND_COLOR] + [CLASS_COLORS[label] for label in sorted(TUMOR_CLASS_NAMES)]
    # Bin edges at half-integers map every integer class id to exactly one color.
    class_boundaries = np.arange(NUM_CLASSES + 1) - 0.5
    return ListedColormap(colors), BoundaryNorm(class_boundaries, NUM_CLASSES)


def load_sample(
    sample: SegmentationSample,
) -> Tuple[NDArray[np.generic], NDArray[np.uint8]]:
    image = read_image_file(sample.image_path, cv2.IMREAD_UNCHANGED)
    mask = read_image_file(sample.mask_path, cv2.IMREAD_GRAYSCALE)
    return image, mask


def describe_sample(
    sample: SegmentationSample, image: NDArray[np.generic], mask: NDArray[np.uint8]
) -> str:
    class_name = TUMOR_CLASS_NAMES[sample.class_label]
    return (
        f"{sample.image_path.name}  |  {class_name} (class {sample.class_label})  |  "
        f"patient {sample.patient_id}\n"
        f"image {image.dtype} {image.shape}, range [{image.min()}, {image.max()}]  |  "
        f"mask values {np.unique(mask).tolist()}, tumor pixels {int(np.count_nonzero(mask))}"
    )


def plot_sample(
    sample: SegmentationSample, image: NDArray[np.generic], mask: NDArray[np.uint8]
) -> Figure:
    colormap, class_norm = build_class_colormap()
    display_image = rescale_min_max(image)
    tumor_only_mask = np.ma.masked_equal(mask, BACKGROUND_LABEL)

    figure, (image_axis, mask_axis, overlay_axis) = plt.subplots(
        1, 3, figsize=(15, 5.8), constrained_layout=True
    )
    image_axis.imshow(display_image, cmap="gray", vmin=0.0, vmax=1.0)
    mask_axis.imshow(mask, cmap=colormap, norm=class_norm, interpolation="nearest")
    overlay_axis.imshow(display_image, cmap="gray", vmin=0.0, vmax=1.0)
    overlay_axis.imshow(
        tumor_only_mask,
        cmap=colormap,
        norm=class_norm,
        alpha=OVERLAY_ALPHA,
        interpolation="nearest",
    )

    for axis, title in (
        (image_axis, "MRI (contrast-scaled)"),
        (mask_axis, "Mask"),
        (overlay_axis, "Overlay"),
    ):
        axis.set_title(title)
        axis.axis("off")

    mask_axis.legend(
        handles=[
            Patch(facecolor=CLASS_COLORS[label], label=f"{label} = {class_name}")
            for label, class_name in TUMOR_CLASS_NAMES.items()
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(TUMOR_CLASS_NAMES),
        frameon=False,
    )
    figure.suptitle(describe_sample(sample, image, mask), fontsize=10)
    return figure


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=(TRAINING_SPLIT, TESTING_SPLIT), default=TRAINING_SPLIT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    samples = discover_samples(arguments.data_dir / arguments.split)
    sample = random.Random(arguments.seed).choice(samples)
    image, mask = load_sample(sample)
    figure = plot_sample(sample, image, mask)

    output_path: Optional[Path] = arguments.output
    if output_path is None:
        plt.show()
        return
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to '{output_path}'.")


if __name__ == "__main__":
    main()
