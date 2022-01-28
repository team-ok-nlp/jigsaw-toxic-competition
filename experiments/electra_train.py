import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import getData, set_seed, format_time
from data import DataProcessor
from model import AutoRegressor_v2
import time
import json

CONFIG = dict(
    seed = 12345,
    pretrained_model = 'google/electra-base-discriminator',
    output_dir = '../models/electra-base-discriminator_regression_v1',
    train_file = '4th/v1/train.csv',
    dev_file = '4th/v1/dev.csv',
    # output_dir = '../models/electra-test',
    # train_file = '../data/4th/v0/tr.csv',
    # dev_file = '../data/4th/v0/dv.csv',
    train_batch_size = 32,
    dev_batch_size = 32,
    lr = 5e-5,
    epochs = 5,
    num_class = 1,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    device_ids = [0,1]
)

def train(config, model, train_dataloader, dev_dataloader, criterion, optimizer, scheduler, epochs, device):

    torch.cuda.empty_cache()
    model.train()
    
    # Measure how long the training epoch takes.
    t0 = time.time()

    best_loss = np.inf
    
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

            # if i!=0 and i%(len(train_dataloader)//3) == 0:
            #     #print(f'Epochs: {epoch_num + 1} | Train Loss: {batch_loss: .3f}')
            #     torch.save(model.module.state_dict(),\
            #             os.path.join(config['output_dir'], f'model_ckpt-{(epoch_num+1)*i}.pt'))

        # validate using our dev set 
        model.eval()
        total_dev_loss = 0.0
        
        with torch.no_grad():
            for dev_input, dev_label in dev_dataloader:
                dev_label = dev_label.to(device)
                input_id = dev_input['input_ids'].squeeze(1).to(device)
                mask = dev_input['attention_mask'].squeeze(1).to(device)

                outputs = model(input_id, mask)
                #print(len(output))
                outputs = torch.squeeze(outputs, 1)

                batch_loss = criterion(outputs.float(), dev_label.float())
                total_dev_loss += batch_loss.item()
                
                del dev_label
                del input_id
                del mask
        
        avg_dev_loss = total_dev_loss / len(dev_dataloader)
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataloader): .3f} \
            | Val Loss: {avg_dev_loss: .3f}')

        if best_loss > avg_dev_loss:
            best_loss = avg_dev_loss

            if not os.path.isdir(config['output_dir']):
                os.mkdir(config['output_dir'])

            torch.save(model.module.state_dict(),\
                    os.path.join(config['output_dir'], f'model_ckpt.pt'))
            # config 파일 확인해서 없으면 저장
            if not os.path.isfile(os.path.join(config['output_dir'],'config.json')):
                # torch.device is not serializable
                del config['device']
                with open(os.path.join(config['output_dir'],'config.json'), 'w') as writer:
                    json.dump(config, writer)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("  Training epoch took: {:}".format(training_time))
    print(f'model ')


if __name__ == '__main__':
    set_seed(CONFIG['seed'])

    # load model
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['pretrained_model'])
    model = AutoRegressor_v2(CONFIG['pretrained_model'], CONFIG['num_class'])

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
    train(CONFIG, model, train_dataloader, dev_dataloader, criterion, optimizer, scheduler, epochs, device)