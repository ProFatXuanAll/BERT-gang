r"""Fine-tune BERT teacher models.

Usage:
    import fine_tune

    model = fine_tune.model.TeacherBert(...)
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

from transformers import BertModel


class TeacherBert(nn.Module):
    r"""Fine-tune BERT model as teacher model.

    Args:
        dropout:
            Dropout probability.
        num_class:
            Number of classes to classify.
        ptrain_ver:
            Pretrained BERT version provided by `transformers` package. See
            https://huggingface.co/transformers/pretrained_models.html for
            details.

    Attributes:
        allow_ptrain_ver:
            Currently supported pre-trained BERT versions. Must be updated when
            adding more supported pre-trained model version. Each supported
            version must indicate its hidden dimension size. See
            https://huggingface.co/transformers/pretrained_models.html for
            details.

    Raises:
        ValueError:
            If `ptrain_ver` is not in `allow_ptrain_ver`
    """

    allow_ptrain_ver = {
        'bert-base-cased': 768,
        'bert-base-uncased': 768,
        'bert-large-cased': 1024,
        'bert-large-uncased': 1024,
        'bert-large-cased-whole-word-masking': 1024,
        'bert-large-uncased-whole-word-masking': 1024,
    }

    def __init__(
            self,
            dropout: float,
            num_class: int,
            ptrain_ver: str,
    ):
        super().__init__()

        # Check if `ptrain_ver` is supported.
        if ptrain_ver not in TeacherBert.allow_ptrain_ver:
            raise ValueError(
                '`ptrain_ver` is not supported.\n' +
                'Supported options:' +
                ''.join(list(map(
                    lambda option: f'\n\t--ptrain_ver {option}',
                    TeacherBert.allow_ptrain_ver.keys()
                )))
            )

        # Load pre-train BERT model.
        self.encoder = BertModel.from_pretrained(ptrain_ver)

        # Dropout layer between encoder and linear layer.
        self.dropout = nn.Dropout(dropout)

        # Linear layer project from `d_model` into `num_class`.
        self.linear_layer = nn.Linear(
            in_features=TeacherBert.allow_ptrain_ver[
                ptrain_ver
            ],
            out_features=num_class
        )

        # Linear layer initialization.
        with torch.no_grad():
            nn.init.normal_(
                self.linear_layer.weight,
                mean=0.0,
                std=0.02
            )
            nn.init.zeros_(self.linear_layer.bias)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor
    ):
        r"""Forward pass.

        We use the following notation for the rest of the context.
            - B: batch size.
            - S: sequence length.
            - C: number of class.

        Args:
            input_ids:
                Batch of input token ids. `input_ids` is a `torch.Tensor` with
                numeric type `torch.int64` and size (B, S).
            attention_mask:
                Batch of input attention masks. `attention_mask` is a
                `torch.Tensor` with numeric type `torch.float32` and size
                (B, S).
            token_type_ids:
                Batch of input token type ids. `token_type_ids` is a
                `torch.Tensor` with numeric type `torch.int64` and size (B, S).

        Returns:
            Unnormalized logits with numeric type `torch.float32` and size
            (B, C).
        """
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = output[1]
        pooled_output = self.dropout(pooled_output)
        return self.linear_layer(pooled_output)

    @torch.no_grad()
    def predict(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor,
            token_type_ids: torch.LongTensor
    ):
        r"""Perform prediction on batch of inputs without gradient.

        This method will neither contruct computational graph not calculate
        gradient. We use the following notation for the rest of the context.
            - B: batch size.
            - S: sequence length.
            - C: number of class.

        Args:
            input_ids:
                Batch of input token ids. `input_ids` is a `torch.Tensor` with
                numeric type `torch.int64` and size (B, S).
            attention_mask:
                Batch of input attention masks. `attention_mask` is a
                `torch.Tensor` with numeric type `torch.float32` and size
                (B, S).
            token_type_ids:
                Batch of input token type ids. `token_type_ids` is a
                `torch.Tensor` with numeric type `torch.int64` and size (B, S).

        Returns:
            Softmax normalized logits with numeric type `torch.float32` and
            size (B, C).
        """
        return F.softmax(
            self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            ),
            dim=-1
        )
