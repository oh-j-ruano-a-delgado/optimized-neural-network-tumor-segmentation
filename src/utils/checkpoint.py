"""Load U-Net training checkpoints portably and rebuild the model they describe.

Checkpoints written by ``scripts/train_baseline.py`` embed the parsed CLI arguments, whose
``output_dir`` is a concrete ``pathlib`` path. A checkpoint saved on Kaggle therefore pickles a
``PosixPath`` (``pathlib._local.PosixPath`` on Python 3.13+), which the safe ``weights_only``
unpickler rejects, which cannot be instantiated on Windows, and whose ``pathlib._local`` module
does not even exist before Python 3.13. Instead of falling back to the unsafe
``weights_only=False``, every concrete path global is mapped to its pure counterpart
(``PurePosixPath`` / ``PureWindowsPath``), which is constructible on any operating system and
Python version, so the same checkpoint loads on Kaggle, Linux and Windows alike.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict, Final, List, Mapping, Tuple

import torch
from torch import Tensor

from ..dataset.transforms import DEFAULT_IMAGE_SIZE
from ..models.unet import UNet

PORTABLE_PATH_GLOBALS: Final[List[Tuple[Callable[..., Any], str]]] = [
    (PurePosixPath, "pathlib.PosixPath"),
    (PureWindowsPath, "pathlib.WindowsPath"),
    (PurePosixPath, "pathlib._local.PosixPath"),
    (PureWindowsPath, "pathlib._local.WindowsPath"),
]
REQUIRED_CHECKPOINT_KEYS: Final[Tuple[str, ...]] = ("model_state_dict", "config")
MODEL_CONFIG_KEYS: Final[Tuple[str, ...]] = ("in_channels", "num_classes", "base_filters", "depth")
HYPERPARAMETER_KEYS: Final[Tuple[str, ...]] = (
    "learning_rate",
    "weight_decay",
    "base_filters",
    "depth",
    "batch_size",
    "epochs",
)


@dataclass(frozen=True)
class TrainingCheckpoint:
    path: Path
    model_state_dict: Mapping[str, Tensor]
    model_config: Mapping[str, int]
    epoch: int
    validation_dice: float
    arguments: Mapping[str, Any]

    @property
    def image_size(self) -> int:
        default_side, _ = DEFAULT_IMAGE_SIZE
        return int(self.arguments.get("image_size", default_side))

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return {key: self.arguments[key] for key in HYPERPARAMETER_KEYS if key in self.arguments}


def read_checkpoint_payload(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
    with torch.serialization.safe_globals(PORTABLE_PATH_GLOBALS):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint '{path}' does not contain a dictionary payload.")
    missing_keys = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in payload]
    if missing_keys:
        raise ValueError(f"Checkpoint '{path}' is missing the keys {missing_keys}.")
    return payload


def load_training_checkpoint(path: Path) -> TrainingCheckpoint:
    payload = read_checkpoint_payload(path)
    model_config = payload["config"]
    missing_config_keys = [key for key in MODEL_CONFIG_KEYS if key not in model_config]
    if missing_config_keys:
        raise ValueError(f"Checkpoint '{path}' config is missing {missing_config_keys}.")
    return TrainingCheckpoint(
        path=path,
        model_state_dict=payload["model_state_dict"],
        model_config={key: int(model_config[key]) for key in MODEL_CONFIG_KEYS},
        epoch=int(payload.get("epoch", -1)),
        validation_dice=float(payload.get("validation_dice", float("nan"))),
        arguments=dict(payload.get("arguments", {})),
    )


def build_model_from_checkpoint(checkpoint: TrainingCheckpoint, device: torch.device) -> UNet:
    model = UNet(**checkpoint.model_config)
    model.load_state_dict(checkpoint.model_state_dict)
    return model.to(device).eval()
