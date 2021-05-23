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

class Angular_Distance(torch.nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-8):
        """Torch Module to calculate angular distance.
        For more detail you can refer to formula (2) from original paper [1].

        Notes
        ----------
        [1] Fu, H., Zhou, S., Yang, Q., Tang, J., Liu, G., Liu, K., &
        Li, X. (2020). LRC-BERT: Latent-representation Contrastive Knowledge
        Distillation for Natural Language Understanding.
        arXiv preprint arXiv:2012.07335.

        Parameters
        ----------
        dim : int, optional
            Dimension where cosine similarity is computed, by default 1
        eps : float, optional
            Small value to avoid division by zero, by default 1e-8
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.cosine_sim = torch.nn.CosineSimilarity(
            dim=self.dim,
            eps=self.eps
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Calculate angular distance.

        Parameters
        ----------
        input1 : torch.Tensor
            Input tensor 1
        input2 : torch.Tensor
            Input tensor 2

        Returns
        -------
        torch.Tensor
            Angular distance tensor
        """
        if input1.shape != input2.shape:
            raise ValueError("Dimension of two input tensor does not match")

        return 1 - self.cosine_sim(input1, input2)

class WSL(torch.nn.Module):
    r"""Weighted Soft Labels loss [1] implementation.
    Our implementation is taken from:
    `bellymonster/Weighted-Soft-Label-Distillation <https://github.com/bellymonster/Weighted-Soft-Label-Distillation>`

    Notes
    ----------
    [1] Zhou, H., Song, L., Chen, J., Zhou, Y., Wang, G., Yuan, J., & Zhang, Q. (2021).
    Rethinking Soft Labels for Knowledge Distillation: A Bias-Variance Tradeoff Perspective.
    arXiv preprint arXiv:2102.00650.

    Parameters
    ----------
    temperature : float
        Softmax temperature
    alpha : float
        Balancing hyperparameter of WSL loss.
    num_class : int
        Number of class of downstream task.
    beta: float, optional
        Hard loss weight by defaul 1.
    """
    def __init__(self, temperature: float, alpha: float, num_class: int, beta: float = 1.0):
        super().__init__()

        self.T = temperature
        self.alpha = alpha
        self.num_class = num_class
        self.beta = beta

        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

        self.hard_loss = torch.nn.CrossEntropyLoss()

    def forward(self, t_logits: torch.Tensor, s_logits: torch.Tensor, label: torch.LongTensor):
        # `B`: batch size
        # `C`: num of class
        # `t_logits`: BxC
        # `s_logits`: BxC
        s_input_for_softmax = s_logits / self.T
        t_input_for_softmax = t_logits / self.T

        t_soft_label = self.softmax(t_input_for_softmax)

        softmax_loss = - torch.sum(
            t_soft_label * self.logsoftmax(s_input_for_softmax),
            1,
            keepdim=True
        )

        t_logits_auto = t_logits.detach()
        s_logits_auto = s_logits.detach()
        log_softmax_s = self.logsoftmax(s_logits_auto)
        log_softmax_t = self.logsoftmax(t_logits_auto)
        one_hot_label = F.one_hot(label, num_classes=self.num_class).float()
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

        focal_weight = softmax_loss_s / softmax_loss_t
        ratio_lower = torch.zeros(1).to(t_logits.device)
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        softmax_loss = focal_weight * softmax_loss

        soft_loss = (self.T ** 2) * torch.mean(softmax_loss)

        hard_loss = self.hard_loss(s_logits, label)

        loss = self.beta * hard_loss + self.alpha * soft_loss

        return loss

def soft_target_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
) -> torch.Tensor:
    r"""Soft-target distillation loss (KL divergence) function.

    Args:
        student_logits:
            Student model's output unnormalized logits with numeric type
            `torch.float32` and size (B, C).
        teacher_logits:
            Teacher model's output unnormalized logits with numeric type
            `torch.float32` and size (B, C).

    Returns:
        Soft-target KL divergence loss.
        Hinton, G. (2014). Distilling the Knowledge in a Neural Network use cross entropy.
        Here we follow recent KD paper, use KL divergence loss.
    """
    soft_loss = torch.nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(student_logits, dim=1),
                    F.softmax(teacher_logits, dim=1)
                )
    return soft_loss


def distill_loss(
        hard_target: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        gamma: float = 0.8,
        alpha: float = 0.2,
        softmax_temp: float = 1.0
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
        gamma (optional):
            loss weight of hard target cross entropy, default 0.8
        alpha (optional):
            loss weight of soft target cross entropy.
        softmax_temp (optional):
            softmax temperature, it will apply to both student and teacher logits.
    Returns:
        Hard-target + soft-target cross-entropy loss. See Hinton, G. (2014).
        Distilling the Knowledge in a Neural Network.
    """

    return (
        gamma * F.cross_entropy(student_logits, hard_target) +
        alpha * soft_target_loss(student_logits / softmax_temp, teacher_logits / softmax_temp) * pow(softmax_temp, 2)
    )

def attention_KL_loss(
        teacher_attn: torch.Tensor,
        student_attn: torch.Tensor,
        gamma: int = 10
) -> torch.Tensor:
    r""" KL divergence loss between teacher's and student's attention head
    We use the following notation for the rest of the context.
        - B: batch size.
        - S: sequence length.
        - A: num of attention heads
    Args:
        teacher_attn:
            attention matrix from one of teacher layer with numeric type
            `torch.float32` and size (B, A, S, S)
        student_attn:
            attention matrix from one of student layer with numeric type
            `torch.float32` and size (B, A, S, S)
        gamma (optional):
            scaling factor of attention KL loss
    Returns:
        KL divergence loss between teacher and student attention heads.
    """
    return gamma * F.kl_div(student_attn, teacher_attn, log_target=True)

def hidden_MSE_loss(
        teacher_hidden: torch.Tensor,
        student_hidden: torch.Tensor,
        mu: int = 100,
        # normalized: bool = False
) -> torch.Tensor:
    r""" MSE loss between teacher's and student's hidden states.
    We use the following notation for the reset of the context.
        - B: batch size.
        - S: sequence size.
        - H: hidden size.
    Args:
        teacher_hidden:
            hidden state from one of teacher layer with numeric type
            `torch.float32` and size (B, S, H)
        student_hidden:
            hidden state from one of student layer with numeric type
            `torch.float32` and size (B, S, H)
        mu (optional):
            scaling factor of hidden MSE loss.
    Returns:
        MSE loss between teacher and student hidden states.
    """
    # if normalized:
    #     student_hidden = F.normalize(student_hidden, dim=2)
    #     teacher_hidden = F.normalize(teacher_hidden, dim=2)
    return mu * F.mse_loss(student_hidden, teacher_hidden)

def token_embedding_cossim_loss(teacher_hidden: torch.Tensor, student_hidden: torch.Tensor, mu: float = 1.0) -> torch.Tensor:
    """Given teacher and student hidden states (BxSxD), calculate token embedding similarity loss.
    We will use `torch.nn.functional.consine_embedding_loss`

    Notes:
    ----------
    B: batch size
    S: sequence length
    D: hidden state dimension

    Parameters
    ----------
    teacher_hidden : torch.Tensor
        teacher hidden states of shape BxSxD
    student_hidden : torch.Tensor
        student hidden states of shape BxSxD
    mu : float, optional
        Weight of cosine_embeddin_loss, by default 1.0

    Returns
    -------
    torch.Tensor
        Scalar tensor of loss.
    """
    if teacher_hidden.shape != student_hidden.shape:
        raise ValueError("Hidden state dimension dosen't match")
    if teacher_hidden.dim() != 3 or student_hidden.dim() != 3:
        raise ValueError("Input hidden state should be a 3-D tensor")

    B, S, _ = teacher_hidden.shape
    y = torch.ones(B*S).to(student_hidden.device)
    return mu * F.cosine_embedding_loss(teacher_hidden.view(B*S, -1), student_hidden.view(B*S, -1), target=y)
