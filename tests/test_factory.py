"""Tests for the model factory in ``src.models.factory``.

The SMP U-Net tests build the real ResNet34 network with ``encoder_weights=None``, so they never
download ImageNet weights and run offline. They check the contract the training loop relies on:
one input channel, one logit map per class at the input resolution, a checkpoint config that
round-trips and stores only the fields of its architecture, legacy configs read as the custom
U-Net, and an encoder/decoder parameter split that covers every parameter exactly once.
Unknown encoders and weights that SMP does not offer must raise ValueError before any download.
"""

import pytest
import torch

from src.models.factory import (
    ModelSettings,
    build_model,
    describe_model,
    model_config,
    resolve_encoder_weights,
    settings_from_model_config,
    split_encoder_parameters,
)
from src.models.unet import UNet

SMP_SETTINGS = ModelSettings(architecture="smp_unet", encoder_name="resnet34")


def test_unet_settings_build_the_custom_unet() -> None:
    model = build_model(ModelSettings(base_filters=4, depth=2))
    assert isinstance(model, UNet)
    assert model(torch.rand(2, 1, 32, 32)).shape == (2, 4, 32, 32)


def test_smp_unet_returns_one_logit_map_per_class() -> None:
    model = build_model(SMP_SETTINGS).eval()
    with torch.no_grad():
        logits = model(torch.rand(2, 1, 64, 64))
    assert logits.shape == (2, 4, 64, 64)
    assert logits.min() < 0.0 or logits.max() > 1.0


def test_smp_unet_first_convolution_takes_one_channel() -> None:
    model = build_model(SMP_SETTINGS)
    assert model.encoder.conv1.weight.shape[1] == 1


@pytest.mark.parametrize(
    "settings", [ModelSettings(base_filters=32, depth=4), SMP_SETTINGS], ids=["unet", "smp_unet"]
)
def test_model_config_round_trips(settings: ModelSettings) -> None:
    assert settings_from_model_config(model_config(settings)) == settings


def test_model_config_stores_only_the_fields_of_its_architecture() -> None:
    assert model_config(SMP_SETTINGS) == {
        "architecture": "smp_unet",
        "in_channels": 1,
        "num_classes": 4,
        "encoder_name": "resnet34",
    }
    assert model_config(ModelSettings(base_filters=32, depth=4)) == {
        "architecture": "unet",
        "in_channels": 1,
        "num_classes": 4,
        "base_filters": 32,
        "depth": 4,
    }


def test_legacy_config_without_architecture_reads_as_unet() -> None:
    legacy_config = {"in_channels": 1, "num_classes": 4, "base_filters": 32, "depth": 4}
    assert settings_from_model_config(legacy_config) == ModelSettings(base_filters=32, depth=4)


def test_incomplete_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="encoder_name"):
        settings_from_model_config({"architecture": "smp_unet", "in_channels": 1, "num_classes": 4})


def test_unknown_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model architecture"):
        build_model(ModelSettings(architecture="transformer"))


def test_unknown_encoder_is_rejected() -> None:
    with pytest.raises(ValueError, match="resnet35"):
        build_model(ModelSettings(architecture="smp_unet", encoder_name="resnet35"))


def test_weights_the_encoder_does_not_offer_are_rejected() -> None:
    with pytest.raises(ValueError, match="instagram"):
        build_model(SMP_SETTINGS, encoder_weights="instagram")


def test_pretrained_weights_are_rejected_for_the_custom_unet() -> None:
    with pytest.raises(ValueError, match="smp_unet"):
        build_model(ModelSettings(base_filters=4, depth=2), encoder_weights="imagenet")


@pytest.mark.parametrize(
    ("architecture", "requested", "expected"),
    [
        ("smp_unet", "imagenet", "imagenet"),
        ("smp_unet", "none", None),
        ("unet", "imagenet", None),
        ("unet", "none", None),
    ],
)
def test_resolve_encoder_weights(architecture: str, requested: str, expected: object) -> None:
    assert resolve_encoder_weights(architecture, requested) == expected


def test_resolve_encoder_weights_rejects_unknown_weights() -> None:
    with pytest.raises(ValueError, match="Unknown encoder weights"):
        resolve_encoder_weights("smp_unet", "noisy-student")


def test_split_encoder_parameters_covers_every_parameter_once() -> None:
    model = build_model(SMP_SETTINGS)
    encoder_parameters, other_parameters = split_encoder_parameters(model)
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    other_ids = {id(parameter) for parameter in other_parameters}
    assert encoder_ids == {id(parameter) for parameter in model.encoder.parameters()}
    assert encoder_ids.isdisjoint(other_ids)
    assert encoder_ids | other_ids == {id(parameter) for parameter in model.parameters()}
    assert other_parameters


def test_split_encoder_parameters_rejects_the_custom_unet() -> None:
    with pytest.raises(ValueError, match="no pretrained encoder"):
        split_encoder_parameters(UNet(base_filters=4, depth=2))


def test_describe_model_names_the_architecture() -> None:
    assert describe_model(SMP_SETTINGS) == "smp_unet (resnet34 encoder)"
    assert describe_model(ModelSettings(base_filters=32, depth=4)) == "unet (32 filters, depth 4)"
