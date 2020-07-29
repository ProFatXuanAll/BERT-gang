r"""Fine-tune ALBERT student models.

Usage:
    import fine_tune

    model = fine_tune.model.StudentAlbert(...)
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

from transformers import AlbertConfig, AlbertModel


class StudentAlbert(nn.Module):
    r"""Fine-tune distillation student model based on ALBERT.

    Args:
        dropout:
            Dropout probability.
        d_emb:
            ALBERT's embedding dimension.
        d_ff:
            ALBERT layers feed forward dimension.
        d_model:
            ALBERT layers hidden dimension.
        max_seq_len:
            Maximum input sequence length.
        num_attention_heads:
            Number of attention heads in ALBERT layers.
        num_class:
            Number of classes to classify.
        num_hidden_layers:
            Number of ALBERT layers.
        type_vocab_size:
            ALBERT's token type embedding range.
        vocab_size:
            Vocabulary size for ALBERT's embedding range.
    """

    def __init__(
            self,
            d_emb: int,
            d_ff: int,
            d_model: int,
            dropout: float,
            max_seq_len: int,
            num_attention_heads: int,
            num_class: int,
            num_hidden_layers: int,
            type_vocab_size: int,
            vocab_size: int,
    ):
        super().__init__()

        # Construct ALBERT model.
        self.encoder = AlbertModel(AlbertConfig(
            attention_probs_dropout_prob=dropout,
            classifier_dropout_prob=dropout,
            embedding_size=d_emb,
            hidden_dropout_prob=dropout,
            hidden_size=d_model,
            initializer_range=0.02,
            inner_group_num=1,
            intermediate_size=d_ff,
            layer_norm_eps=1e-12,
            max_position_embeddings=max_seq_len,
            num_hidden_layers=num_hidden_layers,
            num_hidden_groups=1,
            num_attention_heads=num_attention_heads,
            type_vocab_size=type_vocab_size,
            vocab_size=vocab_size
        ))

        # Dropout layer between encoder and linear layer.
        self.dropout = nn.Dropout(dropout)

        # Linear layer project from `d_model` into `num_class`.
        self.linear_layer = nn.Linear(
            in_features=d_model,
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
