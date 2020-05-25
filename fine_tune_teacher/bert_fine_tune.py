from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from transformers import BertModel, BertTokenizer


class BertFineTuneModel(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 pretrained_version):
        super(BertFineTuneModel, self).__init__()
        self.BertLayer = BertModel.from_pretrained(pretrained_version)
        self.linear_layer = nn.Linear(in_features=in_features,
                                      out_features=out_features)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids):
        output = self.BertLayer(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

        return F.softmax(self.linear_layer(output[1]),
                         dim=1)
