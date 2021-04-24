r"""Implement Supervised Contrative Loss for BERT models.
Our implementation refer to Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
We made some modification to fit our tasks.
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
import torch.nn.functional as F

class SupConLoss(nn.Module):
    r"""Supervised Contrastive Loss module.
        This module implementation refers to
        Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
        We made some modification to fit our tasks.

        Parameters
        ----------
        temperature : float, optional
            Temperature of contrastive loss, by default 0.1
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute Supervised Contrastive loss for model.

        Parameters
        ----------
        features : torch.Tensor
            Unnormalized [CLS] hidden vector of shape: BxD.
        labels : torch.Tensor
            Ground truth label of shape: B.

        Returns
        -------
        torch.Tensor
            A loss scalar tensor.
        """
        if features.dim() != 2:
            raise ValueError("`features` should be a 2-dim tensor with shape: [B,D]")
        if labels.dim() != 1:
            raise ValueError("`labels` should be a 1-dim tensor with shape: [B]")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Num of labels does not match num of features")
        if features.device != labels.device:
            raise ValueError("`features` and `labels` should reside on same device")

        batch_size = features.shape[0]
        device = features.device

        # Normalized [CLS] hidden vector.
        features = F.normalize(features, dim=1)

        # Prepare mask matrix.
        # `logits_mask`: mask out self-contrast.
        # `mask`: mask out different label contrast.
        labels = labels.view(-1,1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1,1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute logits.
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        # For numerical stability: prevent NaN.
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Compute log_prob.
        # Note: log(A/B) = log(A) - log(B).
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Detect `NaN` and replace it to zero.
        if torch.isnan(mean_log_prob_pos).any():
            print('-'*20)
            print(f'features: {features}')
            print(f'labels: {labels}')
            print(f'logits_mask: {logits_mask}')
            print(f'mask: {mask}')
            print(f'log_prob: {log_prob}')
            print(f'exp_logits: {exp_logits}')
            print(mean_log_prob_pos)
            mean_log_prob_pos[torch.isnan(mean_log_prob_pos)] = 0
            print(mean_log_prob_pos)
            raise ValueError("Detect NaN tensor from `supconloss`")

        # Reduction: mean
        loss = -1 * mean_log_prob_pos
        loss = loss.mean()

        return loss
