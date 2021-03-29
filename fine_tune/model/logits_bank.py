r"""Implement a data structure to store output logits of teacher model.
It is designed for a more memory efficient contrastive distillation program.
Let you don't have to load whole teacher model to GPU device.
Logits_bank (L) = NxC
N: dataset size
C: classes of output labels
"""
# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd party modules

import torch
import torch.nn as nn

class Logitsbank(nn.Module):
    """Logits bank object

        Parameters
        ----------
        N : int
            Dataset size
        C : int
            Class of output labels
    """

    def __init__(self, N: int, C: int):
        super(Logitsbank, self).__init__()

        self.N = N
        self.C = C

        self.register_buffer("logitsbank", torch.zeros(self.N, self.C))

    @torch.no_grad()
    def update_logits(self, new: torch.Tensor, index: torch.LongTensor):
        """Update key(s) in logits bank

        Parameters
        ----------
        new : torch.Tensor
            shape: BxC
            B: batch size
            C: classes of a single label
        index : torch.LongTensor
            shape: B
            B: batch size
            index(indices) of updated key(S)
        """
        # Update key(s).
        self.logitsbank.index_copy_(0, index, new)

    def forward(self, index: torch.LongTensor) -> torch.Tensor:
        """Given index(ices) and sample relative logits vectors.

        Parameters
        ----------
        index : torch.LongTensor
            Index(ices), 1D tensor.

        Returns
        -------
        torch.Tensor
            Logits vector(s), shape: KxC.
            K: number of samples.
            C: classes.
        """
        return torch.index_select(self.logitsbank, 0, index).detach()
