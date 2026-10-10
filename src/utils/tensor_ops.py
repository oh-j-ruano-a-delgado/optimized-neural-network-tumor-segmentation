import torch.nn.functional as F
from torch import Tensor


def one_hot_encode(target: Tensor, num_classes: int) -> Tensor:
    """Maps a (B, H, W) class-index tensor to a (B, C, H, W) float mask."""
    return F.one_hot(target, num_classes).movedim(-1, 1).float()
