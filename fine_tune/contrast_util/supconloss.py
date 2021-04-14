r"""Implement Supervised Contrative Loss for BERT models.
Our implementation refer to Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
We make some modification to fit our tasks.
For more implementation details, you can refer to follwing link:
https://github.com/HobbitLong/SupContrast

Notation:
B: batch size.
D: [CLS] hidden dimension.
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd party modules

import torch
import torch.nn as nn

class SupConLoss(nn.Module):

    def __init__(self, temperature: float = 0.1):
        r"""Supervised Contrastive Loss module.
        This module implementation refer to
        Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
        We make some modification to fit our tasks.

        Parameters
        ----------
        temperature : float, optional
            Temperature of contrastive loss, by default 0.1
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute Supervised Contrastive loss for model.

        Parameters
        ----------
        features : torch.Tensor
            Hidden vector of shape: BxD.
        labels : torch.Tensor
            Ground truth label of shape: B.

        Returns
        -------
        torch.Tensor
            A loss scalar tensor.
        """
        raise NotImplementedError("Forward of SupConLoss is not implemented")
