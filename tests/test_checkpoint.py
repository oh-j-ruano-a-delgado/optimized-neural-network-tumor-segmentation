"""Tests for portable checkpoint loading, including checkpoints that pickle foreign path classes.

``save_checkpoint`` writes the legacy Phase 2-6 config (no "architecture" key), so the tests that
use it also guard backward compatibility. The SMP tests write the config through
``model_config`` exactly as ``scripts/train_baseline.py`` does and build the ResNet34 U-Net with
``encoder_weights=None``, so they run offline.
"""

import pathlib
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Tuple, Type

import pytest
import torch
from torch import nn

from src.models.factory import ModelSettings, build_model, model_config
from src.models.unet import UNet
from src.utils.checkpoint import build_model_from_checkpoint, load_training_checkpoint


class ForeignPathReference:
    def __init__(self, path_class: Type[PurePath], parts: Tuple[str, ...]) -> None:
        self.path_class = path_class
        self.parts = parts

    def __reduce__(self) -> Tuple[Any, Tuple[str, ...]]:
        return self.path_class, self.parts


def save_checkpoint(path: Path, model: UNet, output_dir: Any) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "in_channels": model.in_channels,
                "num_classes": model.num_classes,
                "base_filters": model.base_filters,
                "depth": model.depth,
            },
            "epoch": 7,
            "validation_dice": 0.5,
            "arguments": {"learning_rate": 1e-3, "image_size": 64, "output_dir": output_dir},
        },
        path,
    )


@pytest.mark.parametrize(
    ("concrete_class", "expected_class"),
    [(pathlib.PosixPath, PurePosixPath), (pathlib.WindowsPath, PureWindowsPath)],
)
def test_foreign_concrete_paths_load_as_pure_paths(
    tmp_path: Path, concrete_class: Type[PurePath], expected_class: Type[PurePath]
) -> None:
    checkpoint_path = tmp_path / "foreign.pth"
    output_dir = ForeignPathReference(concrete_class, ("runs", "baseline"))
    save_checkpoint(checkpoint_path, UNet(base_filters=4, depth=2), output_dir)

    checkpoint = load_training_checkpoint(checkpoint_path)

    loaded_output_dir = checkpoint.arguments["output_dir"]
    assert type(loaded_output_dir) is expected_class
    assert loaded_output_dir.parts == ("runs", "baseline")


def test_checkpoint_metadata_is_exposed(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "local.pth"
    save_checkpoint(checkpoint_path, UNet(base_filters=4, depth=2), Path("runs") / "local")

    checkpoint = load_training_checkpoint(checkpoint_path)

    assert checkpoint.epoch == 7
    assert checkpoint.validation_dice == pytest.approx(0.5)
    assert checkpoint.image_size == 64
    assert checkpoint.hyperparameters == {"learning_rate": 1e-3}
    assert checkpoint.model_settings == ModelSettings(
        architecture="unet", in_channels=1, num_classes=4, base_filters=4, depth=2
    )


def test_rebuilt_model_reproduces_original_predictions(tmp_path: Path) -> None:
    torch.manual_seed(0)
    original_model = UNet(base_filters=4, depth=2).eval()
    checkpoint_path = tmp_path / "roundtrip.pth"
    save_checkpoint(checkpoint_path, original_model, Path("runs"))

    rebuilt_model = build_model_from_checkpoint(
        load_training_checkpoint(checkpoint_path), torch.device("cpu")
    )

    images = torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        assert torch.allclose(original_model(images), rebuilt_model(images))
    assert not rebuilt_model.training


def save_smp_checkpoint(path: Path, model: nn.Module, settings: ModelSettings) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model_config(settings),
            "epoch": 3,
            "validation_dice": 0.8,
            "arguments": {
                "model": "smp_unet",
                "encoder": "resnet34",
                "encoder_weights": "imagenet",
                "learning_rate": 3e-4,
                "encoder_lr_multiplier": 0.1,
                "base_filters": 16,
                "depth": 4,
            },
        },
        path,
    )


def test_rebuilt_smp_model_reproduces_original_predictions(tmp_path: Path) -> None:
    torch.manual_seed(0)
    settings = ModelSettings(architecture="smp_unet", encoder_name="resnet34")
    original_model = build_model(settings).eval()
    checkpoint_path = tmp_path / "smp.pth"
    save_smp_checkpoint(checkpoint_path, original_model, settings)

    checkpoint = load_training_checkpoint(checkpoint_path)
    rebuilt_model = build_model_from_checkpoint(checkpoint, torch.device("cpu"))

    assert checkpoint.model_settings == settings
    images = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(original_model(images), rebuilt_model(images))
    assert not rebuilt_model.training


def test_smp_hyperparameters_omit_custom_unet_arguments(tmp_path: Path) -> None:
    settings = ModelSettings(architecture="smp_unet", encoder_name="resnet34")
    checkpoint_path = tmp_path / "smp.pth"
    save_smp_checkpoint(checkpoint_path, build_model(settings), settings)

    assert load_training_checkpoint(checkpoint_path).hyperparameters == {
        "model": "smp_unet",
        "encoder": "resnet34",
        "encoder_weights": "imagenet",
        "learning_rate": 3e-4,
        "encoder_lr_multiplier": 0.1,
    }


def test_checkpoint_with_incomplete_smp_config_is_rejected(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "incomplete_smp.pth"
    torch.save(
        {"model_state_dict": {}, "config": {"architecture": "smp_unet", "in_channels": 1}},
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="invalid config"):
        load_training_checkpoint(checkpoint_path)


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_training_checkpoint(tmp_path / "missing.pth")


def test_checkpoint_without_config_is_rejected(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "incomplete.pth"
    torch.save({"model_state_dict": {}}, checkpoint_path)
    with pytest.raises(ValueError, match="config"):
        load_training_checkpoint(checkpoint_path)
