import os

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel

class BertRegressor(nn.Module):

    def __init__(self,model_name, num_class):

        super(BertRegressor, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(768, num_class),
            nn.ReLU())

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        outputs = self.regressor(pooled_output)

        return outputs