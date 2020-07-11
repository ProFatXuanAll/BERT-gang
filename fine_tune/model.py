r"""Fine-tune teacher models.

Usage:
    model = fine_tune.model.TeacherAlbert(...)
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
from transformers import AlbertConfig, AlbertModel, BertConfig, BertModel


class TeacherAlbert(nn.Module):
    r"""Fine-tune ALBERT model as teacher model.

    Args:
        dropout:
            Dropout probability.
        num_class:
            Number of classes to classify.
        pretrained_version:
            Pretrained model provided by hugginface.

    Attributes:
        allow_pretrained_version:
            Currently supported pre-trained ALBERT checkpoints.
            Must be updated when adding more supported pre-trained model.
        pretrain_version_hidden_dim:
            Hidden dimension for each pre-trained ALBERT model.

    Raises:
        ValueError:
            If `pretraind_version` is not in `allow_pretrained_version`
    """

    allow_pretrained_version = [
        'albert-base-v1',
        'albert-base-v2',
    ]

    pretrain_version_hidden_dim = {
        'albert-base-v1': 768,
        'albert-base-v2': 768,
    }

    def __init__(
            self,
            dropout: float,
            num_class: int,
            pretrained_version: str,
    ):
        super(TeacherAlbert, self).__init__()

        if pretrained_version not in TeacherAlbert.allow_pretrained_version:
            raise ValueError(
                '`pretrained_version` is not supported.\n' +
                'supported options:' +
                ''.join(
                    map(
                        lambda option: f'\n\t- "{option}"',
                        TeacherAlbert.allow_pretrained_version
                    )
                )
            )

        self.encoder = AlbertModel.from_pretrained(pretrained_version)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(
            in_features=TeacherAlbert.pretrain_version_hidden_dim[
                pretrained_version
            ],
            out_features=num_class,
            bias=True
        )
        with torch.no_grad():
            nn.init.normal_(
                self.linear_layer.weight,
                mean=0.0,
                std=0.02
            )
            nn.init.zeros_(self.linear_layer.bias)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor,
            token_type_ids: torch.LongTensor
    ):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = output[1]
        pooled_output = self.dropout(pooled_output)
        return F.softmax(
            self.linear_layer(pooled_output),
            dim=1
        )


class TeacherBert(nn.Module):
    r"""Fine-tune BERT model as teacher model.

    Args:
        dropout:
            Dropout probability.
        num_class:
            Number of classes to classify.
        pretrained_version:
            Pretrained model provided by hugginface.

    Attributes:
        allow_pretrained_version:
            Currently supported pre-trained BERT checkpoints.
            Must be updated when adding more supported pre-trained model.
        pretrain_version_hidden_dim:
            Hidden dimension for each pre-trained BERT model.

    Raises:
        ValueError:
            If `pretraind_version` is not in `allow_pretrained_version`
    """

    allow_pretrained_version = (
        'bert-base-cased',
        'bert-base-uncased',
    )

    pretrain_version_hidden_dim = {
        'bert-base-cased': 768,
        'bert-base-uncased': 768,
    }

    def __init__(
            self,
            dropout: float,
            num_class: int,
            pretrained_version: str,
    ):
        super(TeacherBert, self).__init__()

        if pretrained_version not in TeacherBert.allow_pretrained_version:
            raise ValueError(
                '`pretrained_version` is not supported.\n' +
                'supported options:' +
                ''.join(
                    map(
                        lambda option: f'\n\t- "{option}"',
                        TeacherBert.allow_pretrained_version
                    )
                )
            )

        self.encoder = BertModel.from_pretrained(pretrained_version)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(
            in_features=TeacherBert.pretrain_version_hidden_dim[
                pretrained_version
            ],
            out_features=num_class,
            bias=True
        )
        with torch.no_grad():
            nn.init.normal_(
                self.linear_layer.weight,
                mean=0.0,
                std=0.02
            )
            nn.init.zeros_(self.linear_layer.bias)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor,
            token_type_ids: torch.LongTensor
    ):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = output[1]
        pooled_output = self.dropout(pooled_output)
        return F.softmax(
            self.linear_layer(pooled_output),
            dim=1
        )


class StudentAlbert(nn.Module):
    r"""Fine-tune distill ALBERT model as student model.

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
        super(StudentAlbert, self).__init__()

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
        self.dropout = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(
            in_features=d_model,
            out_features=num_class,
            bias=True
        )
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
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = output[1]
        pooled_output = self.dropout(pooled_output)
        return F.softmax(
            self.linear_layer(pooled_output),
            dim=1
        )


class StudentBert(nn.Module):
    r"""Fine-tune distill BERT model as student model.

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
        super(StudentBert, self).__init__()

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
            vocab_size=vocab_size
        ))
        self.dropout = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(
            in_features=d_model,
            out_features=num_class,
            bias=True
        )
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
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = output[1]
        pooled_output = self.dropout(pooled_output)
        return F.softmax(
            self.linear_layer(pooled_output),
            dim=1
        )
