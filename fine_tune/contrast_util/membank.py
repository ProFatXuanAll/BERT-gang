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
    r"""Memory bank object
    """
    def __init__(self,  N: int, dim: int = 768, K: int = 30000, T: float = 0.07):
        """Memory bank constructor

        Parameters
        ----------
        N: int, required
            dataset size
        dim : int, optional
            hidden state dimension, by default 768
        K : int, optional
            number of negative examples, by default 30000
        T : float, optional
            softmax temperature, by default 0.07
        """
        super(Memorybank, self).__init__()

        self.N = N
        self.dim = dim
        self.K = K
        self.T = T

        # create memory bank with random initialization and normalization.
        # memory bank is a tensor with shape DxN.
        # D: hidden state dimension.
        # N: dataset size.
        self.register_buffer("membank", torch.randn(dim, self.N))
        self.membank = nn.functional.normalize(self.membank, dim=0)
