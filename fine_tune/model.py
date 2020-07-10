"""Fine-tune teacher models.

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
from transformers import AlbertModel, BertModel


class TeacherAlbert(nn.Module):
    """Fine-tune ALBERT model as teacher model.

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
                std=1.0
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


class TeacherBert(nn.Module):
    """Fine-tune BERT model as teacher model.

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
                std=1.0
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
