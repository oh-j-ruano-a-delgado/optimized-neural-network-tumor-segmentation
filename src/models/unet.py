from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_IN_CHANNELS: Final[int] = 1
DEFAULT_NUM_CLASSES: Final[int] = 4
DEFAULT_BASE_FILTERS: Final[int] = 16
DEFAULT_DEPTH: Final[int] = 4


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.block(features)


def validate_architecture(num_classes: int, base_filters: int, depth: int) -> None:
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}.")
    if base_filters < 1:
        raise ValueError(f"base_filters must be positive, got {base_filters}.")
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}.")


def initialize_weights(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class UNet(nn.Module):
    """Encoder-decoder U-Net with skip connections whose depth and width are
    hyperparameters, so the genetic algorithm of the optimization phase can
    evolve the architecture without touching this code."""

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        num_classes: int = DEFAULT_NUM_CLASSES,
        base_filters: int = DEFAULT_BASE_FILTERS,
        depth: int = DEFAULT_DEPTH,
    ) -> None:
        super().__init__()
        validate_architecture(num_classes, base_filters, depth)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.depth = depth

        encoder_channels = [base_filters * 2**level for level in range(depth)]
        self.encoder_stages = nn.ModuleList()
        self.pools = nn.ModuleList()
        previous_channels = in_channels
        for channels in encoder_channels:
            self.encoder_stages.append(DoubleConv(previous_channels, channels))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            previous_channels = channels

        self.bottleneck = DoubleConv(previous_channels, encoder_channels[-1] * 2)

        self.up_projections = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        previous_channels = encoder_channels[-1] * 2
        for channels in reversed(encoder_channels):
            self.up_projections.append(nn.Conv2d(previous_channels, channels, kernel_size=1))
            self.decoder_stages.append(DoubleConv(channels * 2, channels))
            previous_channels = channels

        self.classifier = nn.Conv2d(base_filters, num_classes, kernel_size=1)
        self.apply(initialize_weights)

    def forward(self, image: Tensor) -> Tensor:
        skip_features = []
        features = image
        for stage, pool in zip(self.encoder_stages, self.pools):
            features = stage(features)
            skip_features.append(features)
            features = pool(features)

        features = self.bottleneck(features)

        for projection, stage, skip in zip(
            self.up_projections, self.decoder_stages, reversed(skip_features)
        ):
            # Bilinear upsampling with a 1x1 projection avoids the checkerboard
            # artifacts that transposed convolutions can leave in the mask.
            features = F.interpolate(features, scale_factor=2.0, mode="bilinear", align_corners=False)
            features = projection(features)
            if features.shape[-2:] != skip.shape[-2:]:
                raise ValueError(
                    f"Feature map of size {tuple(features.shape[-2:])} does not match skip "
                    f"connection of size {tuple(skip.shape[-2:])}; the input spatial size "
                    f"must be divisible by 2**depth ({2**self.depth})."
                )
            features = stage(torch.cat([features, skip], dim=1))

        return self.classifier(features)
