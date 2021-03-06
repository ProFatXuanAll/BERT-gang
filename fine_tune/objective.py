r"""Fine-tune distillation objective.

Usage:
    loss = soft_target_loss(...)
    loss = distill(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 3rd party modules

import torch
import torch.nn
import torch.nn.functional as F


def soft_target_cross_entropy_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
) -> torch.Tensor:
    r"""Soft-target cross-entropy loss function.

    We use the following notation for the rest of the context.
        - B: batch size.
        - C: number of class.
        - P: teacher model's output softmax probability.
        - P_i: teacher model's output softmax probability on class i.
        - p: teacher model's output unnormalized logits.
        - p_i: teacher model's output unnormalized logits on class i.
        - Q: student model's output softmax probability.
        - Q_i: student model's output softmax probability on class i.
        - q: student model's output unnormalized logits.
        - q_i: student model's output unnormalized logits on class i.

    We first convert logits into prediction using softmax normalization:
        $$
        P_i = \text{softmax}(p_i) = \frac{e^{p_i}}{\sum_{j=1}^C e^{p_j}} \\
        Q_i = \text{softmax}(q_i) = \frac{e^{q_i}}{\sum_{j=1}^C e^{q_j}}
        $$

    Then we calculate soft-target cross-entropy using following formula:
        $$
        -P * \log Q = -\sum_{i = 1}^c P_i * \log Q_i
        $$

    Args:
        student_logits:
            Student model's output unnormalized logits with numeric type
            `torch.float32` and size (B, C).
        teacher_logits:
            Teacher model's output unnormalized logits with numeric type
            `torch.float32` and size (B, C).

    Returns:
        Soft-target cross-entropy loss. See Hinton, G. (2014). Distilling the
        Knowledge in a Neural Network.
    """
    # `p.size == q.size == (B, C)`.
    p = F.softmax(teacher_logits, dim=-1)
    q = F.softmax(student_logits, dim=-1)

    # `loss.size == (B, C)`.
    loss = -p * q.log()

    # `loss.sum(dim=-1).size == (B)` and `loss.sum(dim=-1).mean().size == (1)`.
    return loss.sum(dim=-1).mean()


def distill_loss(
        hard_target: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
) -> torch.Tensor:
    r"""Knowledge distillation loss function.

    We use the following notation for the rest of the context.
        - B: batch size.
        - C: number of class.
        - P: teacher model's output softmax probability.
        - P_i: teacher model's output softmax probability on class i.
        - p: teacher model's output unnormalized logits.
        - p_i: teacher model's output unnormalized logits on class i.
        - Q: student model's output softmax probability.
        - Q_i: student model's output softmax probability on class i.
        - q: student model's output unnormalized logits.
        - q_i: student model's output unnormalized logits on class i.
        - y: Ground truth class index.

    We first convert logits into prediction using softmax normalization:
        $$
        P_i = \text{softmax}(p_i) = \frac{e^{p_i}}{\sum_{j=1}^C e^{p_j}} \\
        Q_i = \text{softmax}(q_i) = \frac{e^{q_i}}{\sum_{j=1}^C e^{q_j}}
        $$

    Then we calculate soft-target cross-entropy using following formula:
        $$
        -P * \log Q = -\sum_{i = 1}^c P_i * \log Q_i
        $$

    Then we calculate hard-target cross-entropy using following formula:
        $$
        -\log Q_y
        $$

    Finally we add hard-target and soft-target together as total loss.

    Args:
        hard_target:
            Actual label for cross-entropy loss with numeric type `torch.int64`
            and size (B).
        student_logits:
            Student model's output unnormalized logits with numeric type
            `torch.float32` and size (B, C).
        teacher_logits:
            Teacher model's output unnormalized logits with numeric type
            `torch.float32` and size (B, C).

    Returns:
        Hard-target + soft-target cross-entropy loss. See Hinton, G. (2014).
        Distilling the Knowledge in a Neural Network.
    """
    return (
        F.cross_entropy(student_logits, hard_target) +
        soft_target_cross_entropy_loss(student_logits, teacher_logits)
    )
