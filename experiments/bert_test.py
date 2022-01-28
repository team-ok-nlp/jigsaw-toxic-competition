import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from utils import getData, clean
from data import DataProcessor
from model import AutoRegressor

CONFIG = dict(
    seed = 12345,
    pretrained_model = 'bert-base-uncased',
    output_dir = '../models/bert_regression_v0',
	finetune_model = 'model_ckpt-55587.pt',
    train_file = '4th/v0/train.csv',
    dev_file = '4th/v0/dev.csv',
    train_batch_size = 32,
    dev_batch_size = 32,
    lr = 5e-5,
    epochs = 5,
    num_class = 1,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    device_ids = [0,1]
)

# # download transformers pretrained model
# tokenizer = AutoTokenizer.from_pretrained(CONFIG['pretrained_model'])
# tokenizer.save_pretrained(os.path.join(CONFIG['output_dir'], 'tokenizer.pt'))
# bert = AutoModel.from_pretrained(CONFIG['pretrained_model'])
# bert.save_pretrained(os.path.join(CONFIG['output_dir'], 'bert.pt'))

# init bert pretrained model
tokenizer = AutoTokenizer.from_pretrained(CONFIG['pretrained_model'])
model = AutoRegressor(CONFIG['pretrained_model'], CONFIG['num_class'])

checkpoint = torch.load(os.path.join(CONFIG['output_dir'], CONFIG['finetune_model']))
model.load_state_dict(checkpoint)
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model, device_ids=CONFIG['device_ids'])
model.to(CONFIG['device'])


# Validation data 
df_val = getData(data_path="4th/validation_cleaned.csv")
# # Test data
# df_sub = getData(data_path="4th/comments_to_score.csv")
# df_sub = clean(df_sub, 'text')

# dataloader
val1_data = DataProcessor(df_val['less_toxic'], tokenizer, is_eval=True)
val1_dataloader = DataLoader(val1_data, batch_size=CONFIG['dev_batch_size'], shuffle=False, num_workers=4)

val2_data = DataProcessor(df_val['more_toxic'], tokenizer, is_eval=True)
val2_dataloader = DataLoader(val2_data, batch_size=CONFIG['dev_batch_size'], shuffle=False, num_workers=4)

def predict(model, val_dataloader, device):

	torch.cuda.empty_cache()
	model.eval()

	outputs = []
	with torch.no_grad():
		for i, val_input in enumerate(tqdm(val_dataloader)):
			input_id = val_input['input_ids'].squeeze(1).to(device)
			mask = val_input['attention_mask'].squeeze(1).to(device)

			output = model(input_id, mask)
			outputs.extend(output.detach().cpu().numpy())

			del input_id
			del mask

	return outputs

device = CONFIG['device']
print('===== predict less toxic =====')
p1 = predict(model, val1_dataloader, device)	
print('===== predict more toxic =====')
p2 = predict(model, val2_dataloader, device)	

p1 = np.asarray(p1)
p2 = np.asarray(p2)
print('===========================')
print(f'\nValidation Accuracy is { np.round((p1 < p2).mean() * 100,2)}\n')
print('===========================')