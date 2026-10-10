import pytest
import torch

from src.dataset.constants import NUM_CLASSES
from src.models.unet import UNet, count_parameters


def test_output_shape_preserves_spatial_dimensions() -> None:
    model = UNet(in_channels=1, num_classes=4, base_filters=4, depth=3)
    images = torch.randn(2, 1, 64, 64)
    logits = model(images)
    assert logits.shape == (2, 4, 64, 64)


def test_default_output_channels_match_dataset_classes() -> None:
    model = UNet()
    assert model.classifier.out_channels == NUM_CLASSES


def test_wider_network_has_more_parameters() -> None:
    narrow = count_parameters(UNet(base_filters=8, depth=3))
    wide = count_parameters(UNet(base_filters=16, depth=3))
    assert wide > narrow


def test_deeper_network_has_more_parameters() -> None:
    shallow = count_parameters(UNet(base_filters=8, depth=2))
    deep = count_parameters(UNet(base_filters=8, depth=3))
    assert deep > shallow


@pytest.mark.parametrize("kwargs", [{"num_classes": 1}, {"base_filters": 0}, {"depth": 0}])
def test_invalid_architecture_arguments_raise_value_error(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        UNet(**kwargs)


def test_non_divisible_input_raises_informative_error() -> None:
    model = UNet(base_filters=4, depth=3)
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 1, 20, 20))


def test_gradients_flow_to_all_parameters() -> None:
    model = UNet(base_filters=4, depth=2)
    logits = model(torch.randn(1, 1, 32, 32))
    logits.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(gradient is not None for gradient in gradients)
