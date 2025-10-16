import torch
import torch.nn as nn

class IdentityProjector(nn.Module):
    """
    A PyTorch module that acts as an identity function.

    It simply returns the input tensor unmodified in the forward pass.
    Used as a placeholder in training students which don't need a projection
    """
    def __init__(self):
        """
        Initializes the IdentityProjector module.

        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass which returns the input tensor unchanged.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The unmodified input tensor.
        """
        return x