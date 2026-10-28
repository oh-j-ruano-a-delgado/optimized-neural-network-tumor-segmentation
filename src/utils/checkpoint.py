"""Load segmentation training checkpoints portably and rebuild the model they describe.

Checkpoints written by ``scripts/train_baseline.py`` embed the parsed CLI arguments, whose
``output_dir`` is a concrete ``pathlib`` path. A checkpoint saved on Kaggle therefore pickles a
``PosixPath`` (``pathlib._local.PosixPath`` on Python 3.13+), which the safe ``weights_only``
unpickler rejects, which cannot be instantiated on Windows, and whose ``pathlib._local`` module
does not even exist before Python 3.13. Instead of falling back to the unsafe
``weights_only=False``, every concrete path global is mapped to its pure counterpart
(``PurePosixPath`` / ``PureWindowsPath``), which is constructible on any operating system and
Python version, so the same checkpoint loads on Kaggle, Linux and Windows alike.

The "config" entry is parsed by ``src/models/factory.py`` into a ``ModelSettings``: either the
custom U-Net or the SMP U-Net with a pretrained encoder. Configs without an "architecture" key
come from the Phase 2-6 runs and describe the custom U-Net. ``build_model_from_checkpoint``
rebuilds the network without downloading pretrained weights, since the trained state dict
replaces them anyway.

``hyperparameters`` lists the stored CLI arguments that describe a run. Arguments that do not
apply to the checkpoint's architecture are left out (``base_filters`` and ``depth`` for the SMP
U-Net, ``encoder`` and ``encoder_weights`` for the custom U-Net), because the training script
records their unused defaults too.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Callable, Dict, Final, List, Mapping, Tuple

import torch
from torch import Tensor, nn

from ..dataset.transforms import DEFAULT_IMAGE_SIZE
from ..models.factory import ModelSettings, build_model, settings_from_model_config

PORTABLE_PATH_GLOBALS: Final[List[Tuple[Callable[..., Any], str]]] = [
    (PurePosixPath, "pathlib.PosixPath"),
    (PureWindowsPath, "pathlib.WindowsPath"),
    (PurePosixPath, "pathlib._local.PosixPath"),
    (PureWindowsPath, "pathlib._local.WindowsPath"),
]
REQUIRED_CHECKPOINT_KEYS: Final[Tuple[str, ...]] = ("model_state_dict", "config")
HYPERPARAMETER_KEYS: Final[Tuple[str, ...]] = (
    "model",
    "encoder",
    "encoder_weights",
    "learning_rate",
    "encoder_lr_multiplier",
    "weight_decay",
    "base_filters",
    "depth",
    "batch_size",
    "epochs",
)
UNUSED_HYPERPARAMETER_KEYS: Final[Mapping[str, Tuple[str, ...]]] = MappingProxyType(
    {
        "unet": ("encoder", "encoder_weights"),
        "smp_unet": ("base_filters", "depth"),
    }
)


@dataclass(frozen=True)
class TrainingCheckpoint:
    path: Path
    model_state_dict: Mapping[str, Tensor]
    model_settings: ModelSettings
    epoch: int
    validation_dice: float
    arguments: Mapping[str, Any]

    @property
    def image_size(self) -> int:
        default_side, _ = DEFAULT_IMAGE_SIZE
        return int(self.arguments.get("image_size", default_side))

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        unused_keys = UNUSED_HYPERPARAMETER_KEYS[self.model_settings.architecture]
        return {
            key: self.arguments[key]
            for key in HYPERPARAMETER_KEYS
            if key in self.arguments and key not in unused_keys
        }


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
    try:
        model_settings = settings_from_model_config(payload["config"])
    except ValueError as error:
        raise ValueError(f"Checkpoint '{path}' has an invalid config: {error}") from error
    return TrainingCheckpoint(
        path=path,
        model_state_dict=payload["model_state_dict"],
        model_settings=model_settings,
        epoch=int(payload.get("epoch", -1)),
        validation_dice=float(payload.get("validation_dice", float("nan"))),
        arguments=dict(payload.get("arguments", {})),
    )


def build_model_from_checkpoint(checkpoint: TrainingCheckpoint, device: torch.device) -> nn.Module:
    model = build_model(checkpoint.model_settings)
    model.load_state_dict(checkpoint.model_state_dict)
    return model.to(device).eval()
