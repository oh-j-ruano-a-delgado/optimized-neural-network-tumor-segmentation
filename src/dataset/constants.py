from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
FIGSHARE_ARCHIVE_DIR: Final[Path] = RAW_DATA_DIR / "figshare"

TRAINING_SPLIT: Final[str] = "Training"
TESTING_SPLIT: Final[str] = "Testing"
IMAGES_DIRNAME: Final[str] = "images"
MASKS_DIRNAME: Final[str] = "masks"
SAMPLE_NAME_SEPARATOR: Final[str] = "_"

BACKGROUND_LABEL: Final[int] = 0
# Figshare already encodes the tumor type as 1..3, so reusing those ids as mask
# pixel values avoids any remapping and leaves 0 free for the background.
TUMOR_CLASS_NAMES: Final[Mapping[int, str]] = MappingProxyType(
    {1: "meningioma", 2: "glioma", 3: "pituitary"}
)
NUM_CLASSES: Final[int] = len(TUMOR_CLASS_NAMES) + 1


def build_sample_name(patient_id: str, record_id: str) -> str:
    return f"{patient_id}{SAMPLE_NAME_SEPARATOR}{record_id}.png"


def parse_patient_id(sample_path: Path) -> str:
    return sample_path.stem.split(SAMPLE_NAME_SEPARATOR, 1)[0]
