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
    device_id : int, optional
        device id of memory bank, by default -1
    """
    def __init__(self,  N: int, dim: int, device_id: int = -1):
        super(Memorybank, self).__init__()

        self.N = N
        self.dim = dim
        self.device_id = device_id
        # self.K = K

        # create memory bank with random initialization and normalization.
        # memory bank is a tensor with shape DxN.
        # D: hidden state dimension.
        # N: dataset size.
        self.register_buffer("membank", torch.randn(dim, self.N))
        self.membank = nn.functional.normalize(self.membank, dim=0)

    @property
    def device(self) -> torch.device:
        r"""Get memory bank device.

        Returns
        -------
        torch.device
            Device create by `torch.device`
        """
        if self.device_id == -1:
            return torch.device('cpu')
        return torch.device(f'cuda:{self.device_id}')

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

    def forward(self, n_index: torch.LongTensor) -> torch.Tensor:
        r"""Given negative sample(s) index(ices) and return negative sample(s).

        Parameters
        ----------
        n_index : torch.LongTensor
            negative sample(s) index(ices), 1D tensor.

        Returns
        -------
        torch.Tensor
            negative sample(s), shape: DxK.
            D: dimension
            K: number of negative samples
        """
        return torch.index_select(self.membank, 1, n_index).detach()
