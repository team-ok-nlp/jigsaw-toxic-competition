import os

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel

class AutoRegressor(nn.Module):

    def __init__(self,model_name, num_class):

        super(AutoRegressor, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(768, num_class))

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        outputs = self.regressor(pooled_output)

        return outputs