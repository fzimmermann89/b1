"""Utility functions for B1 prediction."""

import torch
from einops import rearrange


def complex_to_real(x: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor to real tensor.

    Parameters
    ----------
    x
        Complex tensor

    Returns
    -------
    torch.Tensor
        Real tensor
    """
    return rearrange(torch.view_as_real(x), 'b c ... realimag -> b (c realimag) ...')


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert real tensor to complex tensor.

    Parameters
    ----------
    x
        Real tensor

    Returns
    -------
    torch.Tensor
        Complex tensor
    """
    return torch.view_as_complex(rearrange(x, 'b (c realimag) ... -> b c ... realimag', realimag=2).contiguous())
