r"""Fine-tune BERT student models.

Usage:
    import fine_tune

    model = fine_tune.model.StudentBert(...)
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

from transformers import BertConfig, BertModel


class StudentBert(nn.Module):
    r"""Fine-tune distillation student model based on BERT.

    Args:
        dropout:
            Dropout probability.
        d_ff:
            BERT layers feed forward dimension.
        d_model:
            BERT layers hidden dimension.
        max_seq_len:
            Maximum input sequence length.
        num_attention_heads:
            Number of attention heads in BERT layers.
        num_class:
            Number of classes to classify.
        num_hidden_layers:
            Number of BERT layers.
        type_vocab_size:
            BERT's token type embedding range.
        vocab_size:
            Vocabulary size for BERT's embedding range.
    """

    def __init__(
            self,
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

        # Construct BERT model.
        self.encoder = BertModel(BertConfig(
            attention_probs_dropout_prob=dropout,
            gradient_checkpointing=False,
            hidden_dropout_prob=dropout,
            hidden_size=d_model,
            initializer_range=0.02,
            intermediate_size=d_ff,
            layer_norm_eps=1e-12,
            max_position_embeddings=max_seq_len,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            type_vocab_size=type_vocab_size,
            vocab_size=vocab_size,
            return_dict=True
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
            token_type_ids: torch.Tensor,
            return_hidden_and_attn: bool = False
    ):
        r"""Forward pass

        We use the following notation for the rest of the context.
            - A: num of attention heads.
            - B: batch size.
            - S: sequence length.
            - C: number of class.
            - H: hidden state size.

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
            return_hidden_and_attn:
                A boolean flag to indicate whether return hidden states and attention heads
                of model. It should be true if you want to get hidden states of a fine-tuned model.
                Default: `False`

        Returns:
            If `return_hidden_and_attn` is `False`:
                Unnormalized logits with numeric type `torch.float32` and size
                (B, C).
            Else:
                Return three values:
                1. Unnormalized logits with numeric type `torch.float32` and size
                (B, C).
                2. Hidden states: Tuple of torch.FloatTensor with shape: (B, S, H).
                (One for the output of the embeddings + one for the output of each layer.)
                3. Attentions: Tuple of torch.FloatTensor with shape: (B, A, S, S).
                (One for each layer).
        """
        # Return logits, hidden states and attention heads.
        if return_hidden_and_attn:
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                output_attentions=True
            )

            pooled_output = output.pooler_output
            hidden_states = output.hidden_states
            attentions = output.attentions

            pooled_output = self.dropout(pooled_output)
            return self.linear_layer(pooled_output), hidden_states, attentions

        # Only return logits.
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = output.pooler_output
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
                token_type_ids=token_type_ids,
                return_hidden_and_attn=False
            ),
            dim=-1
        )
