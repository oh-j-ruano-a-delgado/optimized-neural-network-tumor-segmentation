from pathlib import Path
from typing import Final, FrozenSet, Optional, Tuple

DATASET_ID: Final[str] = "masoudnickparvar/brain-tumor-mri-dataset"
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
REQUIRED_SPLITS: Final[Tuple[str, ...]] = ("training", "testing")
IMAGE_EXTENSIONS: Final[FrozenSet[str]] = frozenset({".jpg", ".jpeg", ".png"})


def find_split_directory(raw_data_dir: Path, split_name: str) -> Optional[Path]:
    if not raw_data_dir.is_dir():
        return None
    # The Kaggle archive ships "Training"/"Testing" capitalised, so the match is
    # case-insensitive to behave identically on Windows and on Linux/macOS.
    return next(
        (
            candidate
            for candidate in raw_data_dir.iterdir()
            if candidate.is_dir() and candidate.name.lower() == split_name
        ),
        None,
    )


def contains_images(directory: Path) -> bool:
    return any(
        file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
        for file_path in directory.rglob("*")
    )


def is_dataset_available(raw_data_dir: Path) -> bool:
    split_directories = [
        find_split_directory(raw_data_dir, split_name) for split_name in REQUIRED_SPLITS
    ]
    return all(
        split_directory is not None and contains_images(split_directory)
        for split_directory in split_directories
    )


def download_dataset(dataset_id: str, destination: Path) -> None:
    # The kaggle package authenticates at import time, so importing it lazily lets
    # the availability check run on machines without Kaggle credentials.
    from kaggle.api.kaggle_api_extended import KaggleApi

    destination.mkdir(parents=True, exist_ok=True)
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    kaggle_api.dataset_download_files(
        dataset_id, path=str(destination), unzip=True, quiet=False
    )


def main() -> None:
    if is_dataset_available(RAW_DATA_DIR):
        print(f"Dataset already available in '{RAW_DATA_DIR}'. Skipping download.")
        return

    print(f"Downloading '{DATASET_ID}' into '{RAW_DATA_DIR}'...")
    download_dataset(DATASET_ID, RAW_DATA_DIR)
    print("Download completed successfully.")


if __name__ == "__main__":
    main()
