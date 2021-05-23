r"""Implement a memory bank which stores hidden states of teacher model
w.r.t each data sample of whole datasets.
Memory bank (M) = DxN
D: hidden states dimension
N: dataset size
"""
# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd party modules

import torch
import torch.nn as nn


class Memorybank(nn.Module):
    r"""Memory bank object.

    Parameters
    ----------
    N : int
        dataset size
    dim : int
        hidden state dimension
    embd_type : str, optional
        embedding type of memory bank.
        `cls` means that memory bank store [CLS] embedding.
        `mean` means that memory bank store average BERT output embedding.
    """
    def __init__(self,  N: int, dim: int, embd_type: str = 'cls'):
        super(Memorybank, self).__init__()

        self.N = N
        self.dim = dim

        if embd_type.lower() != 'cls' and embd_type.lower() != 'mean':
            raise ValueError(f"Invalid embeddding type:{embd_type}")
        self.embd_type = embd_type.lower()

        # create memory bank with random initialization and normalization.
        # memory bank is a tensor with shape DxN.
        # D: hidden state dimension.
        # N: dataset size.
        self.register_buffer("membank", torch.randn(dim, self.N))
        self.membank = nn.functional.normalize(self.membank, dim=0)

    @torch.no_grad()
    def update_memory(self, new: torch.Tensor, index: torch.LongTensor):
        r"""Update key(s) in memory bank

        Parameters
        ----------
        new : torch.Tensor
            shape: BxD
            B: batch size
            D: dimension of a single key
            new key value(s) to update
        index : torch.LongTensor
            shape: B
            B: batch size
            index(indices) of updated key(s)
        """
        # Normalize new key(s).
        new = nn.functional.normalize(new)

        # Update normalized key(s).
        self.membank.index_copy_(1, index, new.T)

    def forward(self, index: torch.LongTensor) -> torch.Tensor:
        r"""Given negative sample(s) index(ices) and return sample(s).

        Parameters
        ----------
        index : torch.LongTensor
            Sample(s) index(ices), 1D tensor.

        Returns
        -------
        torch.Tensor
            Sample(s), shape: DxK.
            D: dimension
            K: number of samples
        """
        return torch.index_select(self.membank, -1, index).detach()
