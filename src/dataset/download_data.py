import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Sequence, Tuple

import cv2
import h5py
import numpy as np
from numpy.typing import NDArray

from .constants import (
    FIGSHARE_ARCHIVE_DIR,
    IMAGES_DIRNAME,
    MASKS_DIRNAME,
    RAW_DATA_DIR,
    TESTING_SPLIT,
    TRAINING_SPLIT,
    TUMOR_CLASS_NAMES,
    build_sample_name,
)
from .splits import split_patients_stratified

DATASET_ID: Final[str] = "ashkhagan/figshare-brain-tumor-dataset"
TEST_FRACTION: Final[float] = 0.2
SPLIT_SEED: Final[int] = 42
UNKNOWN_PATIENT_ID: Final[str] = "unknown"


@dataclass(frozen=True)
class FigshareRecord:
    mat_path: Path
    label: int
    patient_id: str


def decode_matlab_string(character_codes: NDArray[np.generic]) -> str:
    return "".join(chr(int(code)) for code in character_codes.ravel())


def sanitize_patient_id(raw_patient_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", raw_patient_id) or UNKNOWN_PATIENT_ID


def find_record_files(archive_dir: Path) -> List[Path]:
    # The archive also ships "cvind.mat" (cross-validation indices); only the
    # numerically named files contain image records.
    return sorted(
        (path for path in archive_dir.rglob("*.mat") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


def read_record_metadata(mat_path: Path) -> FigshareRecord:
    with h5py.File(mat_path, "r") as mat_file:
        cjdata = mat_file["cjdata"]
        label = int(np.asarray(cjdata["label"]).item())
        raw_patient_id = decode_matlab_string(np.asarray(cjdata["PID"]))
    return FigshareRecord(mat_path, label, sanitize_patient_id(raw_patient_id))


def read_image_and_tumor_mask(
    mat_path: Path,
) -> Tuple[NDArray[np.uint16], NDArray[np.uint8]]:
    with h5py.File(mat_path, "r") as mat_file:
        cjdata = mat_file["cjdata"]
        # MATLAB v7.3 stores matrices column-major, so HDF5 exposes them transposed.
        image = np.asarray(cjdata["image"]).T
        tumor_mask = np.asarray(cjdata["tumorMask"]).T
    # Raw intensities exceed 8 bits, so they are kept as 16-bit to avoid
    # quantization before normalization.
    image_uint16 = np.clip(image, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return image_uint16, (tumor_mask > 0).astype(np.uint8)


def encode_class_mask(tumor_mask: NDArray[np.uint8], label: int) -> NDArray[np.uint8]:
    return tumor_mask * np.uint8(label)


def write_png(image: NDArray[np.generic], destination: Path) -> None:
    # cv2.imwrite fails silently on Windows paths with non-ASCII characters
    # (e.g. "3º"), so the PNG is encoded in memory and written through NumPy.
    is_encoded, encoded_png = cv2.imencode(".png", image)
    if not is_encoded:
        raise OSError(f"Could not encode PNG for '{destination}'.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded_png.tofile(destination)


def export_record(record: FigshareRecord, split_dir: Path) -> None:
    image, tumor_mask = read_image_and_tumor_mask(record.mat_path)
    class_dir = split_dir / TUMOR_CLASS_NAMES[record.label]
    sample_name = build_sample_name(record.patient_id, record.mat_path.stem)
    write_png(image, class_dir / IMAGES_DIRNAME / sample_name)
    write_png(encode_class_mask(tumor_mask, record.label), class_dir / MASKS_DIRNAME / sample_name)


def assign_splits(
    records: Sequence[FigshareRecord], test_fraction: float, seed: int
) -> Dict[FigshareRecord, str]:
    patient_labels = {record.patient_id: record.label for record in records}
    training_patients, _ = split_patients_stratified(patient_labels, test_fraction, seed)
    return {
        record: TRAINING_SPLIT if record.patient_id in training_patients else TESTING_SPLIT
        for record in records
    }


def is_split_prepared(split_dir: Path) -> bool:
    return any(split_dir.glob(f"*/{MASKS_DIRNAME}/*.png"))


def is_dataset_prepared(raw_data_dir: Path) -> bool:
    return all(
        is_split_prepared(raw_data_dir / split_name)
        for split_name in (TRAINING_SPLIT, TESTING_SPLIT)
    )


def find_populated_split_directories(raw_data_dir: Path) -> List[Path]:
    return [
        split_dir
        for split_dir in (raw_data_dir / TRAINING_SPLIT, raw_data_dir / TESTING_SPLIT)
        if split_dir.is_dir() and any(split_dir.iterdir())
    ]


def download_dataset(dataset_id: str, destination: Path) -> None:
    # The kaggle package authenticates at import time, so importing it lazily lets
    # the availability check run on machines without Kaggle credentials.
    from kaggle.api.kaggle_api_extended import KaggleApi

    destination.mkdir(parents=True, exist_ok=True)
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    kaggle_api.dataset_download_files(dataset_id, path=str(destination), unzip=True, quiet=False)


def prepare_dataset(raw_data_dir: Path, archive_dir: Path) -> None:
    populated_split_dirs = find_populated_split_directories(raw_data_dir)
    if populated_split_dirs:
        stale_paths = ", ".join(f"'{path}'" for path in populated_split_dirs)
        raise RuntimeError(
            f"Found non-segmentation or partially converted data in {stale_paths}. "
            "Delete these folders and run the script again."
        )

    if not find_record_files(archive_dir):
        print(f"Downloading '{DATASET_ID}' into '{archive_dir}'...")
        download_dataset(DATASET_ID, archive_dir)

    records = [read_record_metadata(mat_path) for mat_path in find_record_files(archive_dir)]
    split_assignments = assign_splits(records, TEST_FRACTION, SPLIT_SEED)

    print(f"Converting {len(records)} records into '{raw_data_dir}'...")
    for record, split_name in split_assignments.items():
        export_record(record, raw_data_dir / split_name)

    for split_name, sample_count in sorted(Counter(split_assignments.values()).items()):
        print(f"  {split_name}: {sample_count} samples")


def main() -> None:
    if is_dataset_prepared(RAW_DATA_DIR):
        print(f"Segmentation dataset already available in '{RAW_DATA_DIR}'. Skipping.")
        return

    prepare_dataset(RAW_DATA_DIR, FIGSHARE_ARCHIVE_DIR)
    print("Dataset prepared successfully.")


if __name__ == "__main__":
    main()
