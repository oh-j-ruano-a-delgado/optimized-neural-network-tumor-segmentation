"""Export a training checkpoint with fp16 weights for the Streamlit app.

The app only runs inference, so half-precision weights halve the repository footprint
(Run F: 93 MB -> about 47 MB) while keeping every checkpoint key. The loader casts the
weights back to fp32 before ``load_state_dict``, so inference stays numerically stable
on CPU.

Usage (from the project root):
    python -m scripts.export_checkpoint_fp16 \
        --checkpoint runs/transfer_learning/13_F_smp_unet_resnet34/best_model.pth \
        --output app/models/f_smp_unet_fp16.pth
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Final

import torch
from torch import Tensor

from src.utils.checkpoint import read_checkpoint_payload

DEFAULT_CHECKPOINT: Final[Path] = Path(
    "runs/transfer_learning/13_F_smp_unet_resnet34/best_model.pth"
)
DEFAULT_OUTPUT: Final[Path] = Path("app/models/f_smp_unet_fp16.pth")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def cast_to_half(tensor: Tensor) -> Tensor:
    return tensor.half() if tensor.is_floating_point() else tensor


def to_half_precision(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload["model_state_dict"] = {
        name: cast_to_half(tensor)
        for name, tensor in payload["model_state_dict"].items()
    }
    return payload


def describe_size(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB"


def main() -> None:
    arguments = parse_arguments()
    payload = to_half_precision(read_checkpoint_payload(arguments.checkpoint))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, arguments.output)
    print(
        f"Exported {len(payload['model_state_dict'])} tensors to '{arguments.output}' "
        f"({describe_size(arguments.checkpoint)} -> {describe_size(arguments.output)})."
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
