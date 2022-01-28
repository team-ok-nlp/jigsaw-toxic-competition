import torch
import torch.nn as nn

from transformers import AutoModel

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

class AutoRegressor_v2(nn.Module):
    '''for Electra
    
    '''

    def __init__(self,model_name, num_class):

        super(AutoRegressor_v2, self).__init__()

        self.base = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(768, num_class))
        

    def forward(self, input_id, mask):
        try:
            # output 이 (tensor,) 형태임 -> output[0]
            output = self.base(input_ids=input_id, attention_mask=mask, return_dict=False)[0]

            # pooled_output size : (32, 768)
            pooled_output = torch.squeeze(output[:, 0:1, :])
            
            # outputs size : (32, 1)
            outputs = self.regressor(pooled_output)
            
        except Exception as e:
            print(f'error : {e}')
            # print(f'pooled_output : {pooled_output}')
            # print(f'type : {type(pooled_output)}\n')
            raise Exception('stop training')

        return outputs
