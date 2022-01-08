# %% [markdown]
# # BERT test

# %% [markdown]
# ## 0. import libs

# %%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from parallel import DataParallelModel, DataParallelCriterion

from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import clean, getData

# %% [markdown]
# ## 1. Data Processing

# %%
class DataProcessor(Dataset):
    """
    get batch data and tokenizing
    """

    def __init__(self, df, tokenizer, is_eval=False) -> None:

        self.df = df
        self.tokenizer = tokenizer

        if is_eval:
            self.df = clean(self.df)

    def tokenize(self, text):
        # tokenize and to tensor
        try:
            embedding = self.tokenizer.encode_plus(
                                    text=text,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt"
                        )
        except:
            print('text error')
            print(text)
        
        return embedding

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        inputs = self.df['comment'].values[idx]
        labels = self.df['score'].values[idx]
        # print(inputs)
        # print(labels)

        inputs = self.tokenize(inputs)

        return inputs, labels

# %% [markdown]
# ## 2. BERT Modeling

# %%
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

# %% [markdown]
# ## 3. Training

# %%
# set cofig
CONFIG = dict(
    seed = 12345,
    pretrained_model = 'bert-base-uncased',
    output_dir = '../models/bert_regression_test',
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

# %%
def train(model, epochs, train_dataloader, dev_dataloader, criterion, optimizer, scheduler, device, output_dir):

    torch.cuda.empty_cache()
    model.train()
    for epoch_num in range(epochs):

        total_loss_train = 0.0
        print(f"[Epochs : {epoch_num+1}/{epochs}]")
        for i, (train_input, train_label) in enumerate(tqdm(train_dataloader)):
            input_id = train_input['input_ids'].squeeze(1).to(device)
            mask = train_input['attention_mask'].squeeze(1).to(device)

            output = model(input_id, mask)
            output = torch.squeeze(output, 1)
            del input_id
            del mask
            
            train_label = train_label.to(device)
            batch_loss = criterion(output.float(), train_label.float())
            del train_label

            total_loss_train += batch_loss.item()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()

            if i%1000 == 0:  
                print(f'Epochs: {epoch_num + 1} | Train Loss: {batch_loss: .3f}')

        # validate using our dev set 
        model.eval()
        total_loss_dev = 0.0

        with torch.no_grad():
            for dev_input, dev_label in dev_dataloader:
                dev_label = dev_label.to(device)
                input_id = dev_input['input_ids'].squeeze(1).to(device)
                mask = dev_input['attention_mask'].squeeze(1).to(device)

                output = model(input_id, mask)
                #print(len(output))
                output = torch.squeeze(output, 1)

                batch_loss = criterion(output.float(), dev_label.float())
                total_loss_dev += batch_loss.item()

                del dev_label
                del input_id
                del mask
        
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataloader): .3f} \
            | Val Loss: {total_loss_dev / len(dev_dataloader): .3f}')

        model.save_pretrained(output_dir)

# %% [markdown]
# ### load dataset

# %%
# init bert pretrained model
tokenizer = AutoTokenizer.from_pretrained(CONFIG['pretrained_model'])

# load dataset
# train dataset
train_df = getData(data_path=CONFIG['train_file'])
# data processing with tokenizing
train_data = DataProcessor(train_df, tokenizer, is_eval=False)
train_dataloader = DataLoader(train_data, batch_size=CONFIG['train_batch_size'], shuffle=True, num_workers=0)

# dev dataset
dev_df = getData(data_path=CONFIG['dev_file'])
# data processing with tokenizing
dev_data = DataProcessor(dev_df, tokenizer, is_eval=False)
dev_dataloader = DataLoader(dev_data, batch_size=CONFIG['dev_batch_size'], shuffle=True, num_workers=0)

# %% [markdown]
# ### load pretrained model

# %%
model = BertRegressor(CONFIG['pretrained_model'], CONFIG['num_class'])

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=CONFIG['device_ids'])
    #model = DataParallelModel(model, device_ids=CONFIG['device_ids'])
    model.to(CONFIG['device'])

criterion = nn.MSELoss()
#criterion = DataParallelCriterion(criterion, device_ids=CONFIG['device_ids'])

optimizer = AdamW(
    model.parameters(),
    lr=CONFIG['lr'],
    eps=1e-8,
    correct_bias=False) # AdamW to BERTAdam

epochs = CONFIG['epochs']
num_training_steps = epochs * len(train_dataloader)
num_warmup_steps = 10000
scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0, 
                num_training_steps=num_training_steps
)

# %% [markdown]
# ### fine tuning

# %%
device = CONFIG['device']
output_dir = CONFIG['output_dir']
train(model, epochs, train_dataloader, dev_dataloader, criterion, optimizer, scheduler, device, output_dir)

"""
# %% [markdown]
# ## 4. Validataion and Test

# %% [markdown]
# ### Load Validation and Test data

# %%
# Validation data 
df_val = getData(data_path="4th/validation_data.csv")
# Test data
df_sub = getData(data_path="4th/comments_to_score.csv")
print(df_val.sample(3))
print(df_sub.sample(3))

# %%
df_val = clean(df_val, 'less_toxic')
df_val = clean(df_val, 'more_toxic')
df_sub = clean(df_sub, 'text')

# %% [markdown]
# ## 3. Validation
# - final validation and submission

# %% [markdown]
# ### Analyze bad predictions

# %%
df_val['p1'] = p1
df_val['p2'] = p2
df_val['diff'] = np.abs(p2 - p1)
df_val['correct'] = (p1 < p2).astype('int')

# %%
### Incorrect predictions with similar scores
df_val[df_val.correct == 0].sort_values('diff', ascending=True).head(20)

# %%
### Incorrect predictions with dis-similar scores
df_val[df_val.correct == 0].sort_values('diff', ascending=False).head(20)

# %%


# %% [markdown]
# ## 4. Predict on test data

# %%
# Predict using pipeline
df_sub['score'] = test_preds_arr.mean(axis=1)

# %%
# Cases with duplicates scores
df_sub['score'].count() - df_sub['score'].nunique()

# %%
same_score = df_sub['score'].value_counts().reset_index()[:10]
same_score

# %%
df_sub[df_sub['score'].isin(same_score['index'].tolist())]

# %%
df_sub.sample(5)

# %%
# save submission
df_sub[['comment_id', 'score']].to_csv("submission.csv", index=False)
"""

