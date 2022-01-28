
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import getData
from data import DataProcessor
from torch.utils.data import DataLoader
from model import AutoRegressor_v2

##################################################################
## Validation and Test
##################################################################

def predict(model, val_dataloader, device):

    torch.cuda.empty_cache()
    model.eval()

    outputs = []
    with torch.no_grad():
        for val_input in tqdm(val_dataloader):
            input_id = val_input['input_ids'].squeeze(1).to(device)
            mask = val_input['attention_mask'].squeeze(1).to(device)

            output = model(input_id, mask)
            output = torch.squeeze(output, 1)

            outputs.extend(output.detach().cpu().numpy())

            del input_id
            del mask
            
    return outputs

def validate(config):
    # Validation data 
    df_val = getData(data_path="4th/validation_cleaned.csv")
    
    tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'])

    val1_data = DataProcessor(df_val['less_toxic'], tokenizer, is_eval=True)
    less_dataloader = DataLoader(val1_data, batch_size=config['dev_batch_size'], shuffle=False, num_workers=4)

    val2_data = DataProcessor(df_val['more_toxic'], tokenizer, is_eval=True)
    more_dataloader = DataLoader(val2_data, batch_size=config['dev_batch_size'], shuffle=False, num_workers=4)
    
    # load the model
    # model = AutoModelForSequenceClassification.from_pretrained(config['pretrained_model'], num_labels=config['num_classes'])
    model = AutoRegressor_v2(config['pretrained_model'], num_class=config['num_class'])
    model.load_state_dict(torch.load(os.path.join(config['output_dir'], 'model_ckpt.pt'), map_location=config['device']))

    model.to(config['device'])
    print('processing less toxic data')
    p1 = predict(model, less_dataloader, config['device'])
    print('processing more toxic data')
    p2 = predict(model, more_dataloader, config['device'])

    acc = np.round((np.vstack(p1) < np.vstack(p2)).mean() * 100,2)
    print(f'Validation Accuracy is { acc }')

    with open(os.path.join(config['output_dir'], 'validation_result.txt'), 'w') as writer:
        result_text = f"Model: {config['output_dir']}\nAccuracy: {acc}"
        writer.write(result_text)


if __name__ == '__main__':

    output_dir = '../models/electra-base-discriminator_regression_v1'
    # output_dir = '../models/electra-test'
    # output_dir = '../models/electra-base_regression_mini_fixed'
    # output_dir = '../models/electra-base_regression_test'
    # output_dir = '../models/bert_regression_mini'
    # output_dir = '../models/bert_regression_drop_0.2_mini'
    # output_dir = '../models/bert_regression_drop_0.2_mini_2'
    # output_dir = '../models/bert_regression_drop_0.2_mini_not_tt'
    print(f'model : {output_dir}')

    with open(os.path.join(output_dir, 'config.json'), 'r') as reader:
        config = json.load(reader)
    
    if config['device_ids'][0] >=0 and torch.cuda.is_available():
        config['device'] = torch.device('cuda')

    validate(config)