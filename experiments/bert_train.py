import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import clean, getData
from data import DataProcessor
from model import BertRegressor

CONFIG = dict(
    seed = 12345,
    pretrained_model = 'bert-base-uncased',
    output_dir = '../models/bert_regression_mini',
    train_file = '4th/v0/train_mini.csv',
    dev_file = '4th/v0/dev_mini.csv',
    train_batch_size = 32,
    dev_batch_size = 32,
    lr = 5e-5,
    epochs = 5,
    num_class = 1,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    device_ids = [0,1]
)

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

            if i%10000 == 0:  
                print(f'Epochs: {epoch_num + 1} | Train Loss: {batch_loss: .3f}')
                torch.save(model.state_dict(),\
                        os.path.join(output_dir, f'bert_regression-{epoch_num+1}-{i}.pt'))

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

        torch.save(model.module.state_dict(),\
                os.path.join(output_dir, f'bert_regression-{epoch_num+1}-{len(train_dataloader)}.pt'))


if __name__ == '__main__':
    # load model
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['pretrained_model'])
    model = BertRegressor(CONFIG['pretrained_model'], CONFIG['num_class'])

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=CONFIG['device_ids'])
        model.to(CONFIG['device'])

    # load dataset
    train_df = getData(data_path=CONFIG['train_file'])
    # data processing with tokenizing
    train_data = DataProcessor(train_df, tokenizer, is_eval=False)
    train_dataloader = DataLoader(train_data, batch_size=CONFIG['train_batch_size'], shuffle=True, num_workers=8, pin_memory=True)

    # dev dataset
    dev_df = getData(data_path=CONFIG['dev_file'])
    # data processing with tokenizing
    dev_data = DataProcessor(dev_df, tokenizer, is_eval=False)
    dev_dataloader = DataLoader(dev_data, batch_size=CONFIG['dev_batch_size'], shuffle=True, num_workers=8, pin_memory=True)


    # model config
    criterion = nn.MSELoss()

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        eps=1e-8,
        correct_bias=False) # AdamW to BERTAdam

    epochs = CONFIG['epochs']
    num_training_steps = epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0, 
                    num_training_steps=num_training_steps
    )

    device = CONFIG['device']
    output_dir = CONFIG['output_dir']
    
    train(model, epochs, train_dataloader, dev_dataloader, criterion, optimizer, scheduler, device, output_dir)