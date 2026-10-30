"""Build the segmentation network selected by ``--model`` and describe it in checkpoints.

Both architectures share one interface: an image batch of shape (N, in_channels, H, W) goes in
and raw logits of shape (N, num_classes, H, W) come out, so the losses, metrics and evaluation
scripts work with either of them unchanged.
    "unet"      the from-scratch U-Net of ``src/models/unet.py``, whose width (``base_filters``)
                and depth were tuned by the genetic algorithm (Phases 2-6).
    "smp_unet"  the U-Net of ``segmentation_models_pytorch`` 0.6.0 with a pretrained encoder
                named by ``encoder_name`` (default "resnet34", Phase 7 Run F). It keeps the
                library defaults: encoder depth 5, decoder channels 256-128-64-32-16 with batch
                normalization and nearest-neighbour upsampling, and no output activation, so the
                losses receive logits exactly as they do from the custom U-Net. Its input height
                and width must be divisible by 32, which the 256 x 256 slices are.

Pretrained encoder on grayscale MRI:
    With ``in_channels=1``, SMP keeps the ImageNet weights and replaces the three RGB kernels of
    the first convolution by their sum, so a gray slice x gets the response that the RGB network
    gives to the image (x, x, x). The slices stay min-max normalized to [0, 1] instead of being
    standardized with the ImageNet mean and standard deviation. The first convolution is linear
    and directly followed by batch normalization, so an affine change of the input intensities
    (a * x + b, a > 0) only scales and shifts each of its output channels, which batch
    normalization removes (exactly in the interior, up to the zero padding at the image border).
    Keeping the normalization of Runs A to E also leaves the encoder as the only difference to
    Run E.
    SMP downloads the ImageNet weights from the Hugging Face Hub (``smp-hub/<encoder>.imagenet``)
    when a model is built for training, so that step needs internet access (on Kaggle, turn
    Internet on). Rebuilding a model from a checkpoint never downloads anything: it builds the
    same network with ``encoder_weights=None`` and loads the trained state dict over it.
    Only the encoders that SMP registers by name (for example resnet34, resnet50,
    se_resnext50_32x4d or efficientnet-b3) are accepted. Timm "tu-" encoders take their
    pretrained tag from the name instead of ``encoder_weights`` and are not supported. An unknown
    encoder, or weights that SMP does not offer for it, raise ValueError before any download.

Checkpoint configuration:
    ``model_config`` returns the dictionary stored under "config" in every checkpoint, and
    ``settings_from_model_config`` reverses it. The dictionary holds the architecture and only the
    fields that define it: in_channels, num_classes, base_filters and depth for "unet";
    in_channels, num_classes and encoder_name for "smp_unet". Checkpoints written before Phase 7
    have no "architecture" key and are read as "unet", so every earlier run loads unchanged.
    ``resolve_encoder_weights`` returns None for "unet", so the ``--encoder`` and
    ``--encoder-weights`` options of ``scripts/train_baseline.py`` do not affect the custom U-Net,
    just as ``--base-filters`` and ``--depth`` do not affect "smp_unet".

Encoder learning rate:
    ``split_encoder_parameters`` separates the parameters of the pretrained encoder from those of
    the randomly initialized decoder and segmentation head, so the optimizer can give the encoder
    a smaller learning rate (``--encoder-lr-multiplier`` in ``scripts/train_baseline.py``). The
    custom U-Net has no pretrained encoder, so it raises ValueError.
"""

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Dict, Final, List, Literal, Mapping, Optional, Tuple, cast, get_args

import segmentation_models_pytorch as smp
from torch import nn

from .unet import (
    DEFAULT_BASE_FILTERS,
    DEFAULT_DEPTH,
    DEFAULT_IN_CHANNELS,
    DEFAULT_NUM_CLASSES,
    UNet,
)

ModelArchitecture = Literal["unet", "smp_unet"]
MODEL_ARCHITECTURES: Final[Tuple[str, ...]] = get_args(ModelArchitecture)
DEFAULT_MODEL_ARCHITECTURE: Final[ModelArchitecture] = "unet"
LEGACY_CHECKPOINT_ARCHITECTURE: Final[ModelArchitecture] = "unet"
PRETRAINED_ENCODER_ARCHITECTURE: Final[ModelArchitecture] = "smp_unet"
EncoderWeights = Literal["imagenet", "none"]
ENCODER_WEIGHTS: Final[Tuple[str, ...]] = get_args(EncoderWeights)
DEFAULT_ENCODER_WEIGHTS: Final[EncoderWeights] = "imagenet"
NO_ENCODER_WEIGHTS: Final[EncoderWeights] = "none"
DEFAULT_ENCODER_NAME: Final[str] = "resnet34"
ARCHITECTURE_KEY: Final[str] = "architecture"
ENCODER_NAME_KEY: Final[str] = "encoder_name"
ARCHITECTURE_CONFIG_KEYS: Final[Mapping[str, Tuple[str, ...]]] = MappingProxyType(
    {
        "unet": ("in_channels", "num_classes", "base_filters", "depth"),
        "smp_unet": ("in_channels", "num_classes", ENCODER_NAME_KEY),
    }
)


@dataclass(frozen=True)
class ModelSettings:
    architecture: ModelArchitecture = DEFAULT_MODEL_ARCHITECTURE
    in_channels: int = DEFAULT_IN_CHANNELS
    num_classes: int = DEFAULT_NUM_CLASSES
    base_filters: int = DEFAULT_BASE_FILTERS
    depth: int = DEFAULT_DEPTH
    encoder_name: str = DEFAULT_ENCODER_NAME


def validate_architecture(architecture: str) -> ModelArchitecture:
    if architecture not in MODEL_ARCHITECTURES:
        raise ValueError(
            f"Unknown model architecture '{architecture}', expected one of {MODEL_ARCHITECTURES}."
        )
    return cast(ModelArchitecture, architecture)


def resolve_encoder_weights(architecture: str, requested_weights: str) -> Optional[str]:
    validate_architecture(architecture)
    if requested_weights not in ENCODER_WEIGHTS:
        raise ValueError(
            f"Unknown encoder weights '{requested_weights}', expected one of {ENCODER_WEIGHTS}."
        )
    if architecture != PRETRAINED_ENCODER_ARCHITECTURE or requested_weights == NO_ENCODER_WEIGHTS:
        return None
    return requested_weights


def build_smp_unet(settings: ModelSettings, encoder_weights: Optional[str]) -> nn.Module:
    try:
        return smp.Unet(
            encoder_name=settings.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=settings.in_channels,
            classes=settings.num_classes,
        )
    except KeyError as error:
        raise ValueError(f"Cannot build the SMP U-Net: {error.args[0]}") from error


def build_model(settings: ModelSettings, encoder_weights: Optional[str] = None) -> nn.Module:
    architecture = validate_architecture(settings.architecture)
    if architecture == PRETRAINED_ENCODER_ARCHITECTURE:
        return build_smp_unet(settings, encoder_weights)
    if encoder_weights is not None:
        raise ValueError(
            f"Pretrained encoder weights require the '{PRETRAINED_ENCODER_ARCHITECTURE}' "
            f"architecture, got '{architecture}'."
        )
    return UNet(
        in_channels=settings.in_channels,
        num_classes=settings.num_classes,
        base_filters=settings.base_filters,
        depth=settings.depth,
    )


def model_config(settings: ModelSettings) -> Dict[str, Any]:
    architecture = validate_architecture(settings.architecture)
    field_values = asdict(settings)
    return {
        ARCHITECTURE_KEY: architecture,
        **{key: field_values[key] for key in ARCHITECTURE_CONFIG_KEYS[architecture]},
    }


def settings_from_model_config(config: Mapping[str, Any]) -> ModelSettings:
    architecture = validate_architecture(
        str(config.get(ARCHITECTURE_KEY, LEGACY_CHECKPOINT_ARCHITECTURE))
    )
    required_keys = ARCHITECTURE_CONFIG_KEYS[architecture]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"The '{architecture}' model config is missing {missing_keys}.")
    field_values: Dict[str, Any] = {
        key: str(config[key]) if key == ENCODER_NAME_KEY else int(config[key])
        for key in required_keys
    }
    return ModelSettings(architecture=architecture, **field_values)


def describe_model(settings: ModelSettings) -> str:
    if settings.architecture == PRETRAINED_ENCODER_ARCHITECTURE:
        return f"{settings.architecture} ({settings.encoder_name} encoder)"
    return f"{settings.architecture} ({settings.base_filters} filters, depth {settings.depth})"


def split_encoder_parameters(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError(
            f"{type(model).__name__} has no pretrained encoder to separate; use the "
            f"'{PRETRAINED_ENCODER_ARCHITECTURE}' architecture."
        )
    encoder_parameter_ids = {id(parameter) for parameter in encoder.parameters()}
    encoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) in encoder_parameter_ids
    ]
    other_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_parameter_ids
    ]
    return encoder_parameters, other_parameters
