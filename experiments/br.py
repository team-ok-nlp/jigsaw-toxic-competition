#!/usr/bin/env python
# coding: utf-8

# # BERT Regression 

import pandas as pd
import numpy as np
import random
import torch
import os
import time
import datetime

## Dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
## Dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.nn.utils.clip_grad import clip_grad_norm_
from transformers import AutoModel

from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
from tqdm import tqdm
# from parallel import DataParallelCriterion, DataParallelModel
from utils import clean

# 학습 config는 추후 json 파일로 저장해놓기
CONFIG = dict(
    seed = 12345,
    pretrained_model = 'bert-base-uncased',
    output_dir = '../models/bert_regression_original',
    train_file = '../data/4th/v0/train.csv',
    dev_file = '../data/4th/v0/dev.csv',
    # output_dir = '../models/bert_regression_test',
    # train_file = '../data/4th/v0/tr.csv',
    # dev_file = '../data/4th/v0/dv.csv',
    train_batch_size = 32,
    dev_batch_size = 32,
    lr = 5e-5,
    epochs = 5,
    num_classes = 1,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    device_ids = [0, 1]
)



def set_seed(seed = 12345):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


class RegressionDataset(Dataset):
    '''toxic dataset for BERT regression
    '''
    def __init__(self, tokenizer:AutoTokenizer, file_path, dir_path, mode, force=False) -> None:
        self.file_path = file_path
        self.dir_path = dir_path # output dir
        self.tokenizer = tokenizer
        self.mode = mode
        self.force = force 
        
        # read csv file
        self.data = pd.read_csv(self.file_path)
        self.labels = self.data.score.to_numpy()
        
    def __getitem__(self, idx):
        encodings = self.tokenizer(text=self.data.comment[idx],
                                   padding='max_length',
                                   truncation=True)

        item = {key: torch.tensor(val, dtype=torch.long) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.labels.size
        

class LossEarlyStopper:
    """Early stopper

        patience (int): loss가 줄어들지 않아도 학습할 epoch 수
        patience_counter (int): loss 가 줄어들지 않을 때 마다 1씩 증가
        min_loss (float): 최소 loss
        stop (bool): True 일 때 학습 중단

    """

    def __init__(self, patience: int)-> None:
        # 초기화
        self.patience = patience

        self.patience_counter = 0
        self.min_loss = np.Inf
        self.stop = False

    def check_early_stopping(self, loss: float)-> None:
        if self.min_loss == np.Inf:
            # 첫 에폭
            self.min_loss = loss
            
        elif loss > self.min_loss:
            # loss가 줄지 않음 -> patience_counter 1 증가
            self.patience_counter += 1
            msg = f"Early stopper, Early stopping counter {self.patience_counter}/{self.patience}"

            if self.patience_counter == self.patience:
                # early stop
                self.stop = True

            print(msg)
                
        elif loss <= self.min_loss:
            # loss가 줄어듬 -> min_loss 갱신
            self.save_model = True
            msg = f"Early stopper, Validation loss decreased {self.min_loss} -> {loss}"
            self.min_loss = loss
            self.patience_counter = 0
            print(msg)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def r2_score(outputs, labels):
    '''MSE score
    '''
    labels_mean = np.mean(labels)
    ss_tot = np.sum((labels - labels_mean) ** 2)
    ss_res = np.sum((labels - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

class LMRegressor(nn.Module):

    def __init__(self,model_name, num_class):

        super(LMRegressor, self).__init__()

        self.LM = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(768, num_class),
            nn.ReLU())

    def forward(self, x):

        _, pooled_output = self.LM(input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'], return_dict=False)
        outputs = self.regressor(pooled_output)

        return outputs


def train(config, model, optimizer, scheduler, loss_function, epochs,       
          train_dataloader, dev_dataloader, device='cpu', clip_value=2, freq=40, patience=-1):
    """
    train model
    Arguments:
        
    """
    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()
    best_score = -99999
    
    if patience>-1:
        es = LossEarlyStopper(patience)
    for epoch_i in range(epochs):
        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)): 
            
            batch = {k:v.to(device) for k, v in batch.items()}
#             batch_inputs, batch_masks, batch_labels = \
#                                tuple(b.to(device) for b in batch)
            model.zero_grad()
            outputs = model(**batch)           

            loss = loss_function(outputs.logits.squeeze().float(), batch['labels'].float())
            total_train_loss += loss.item()

            loss.backward()
        
            # Clip the norm of the gradients to clip_value.
            # This is to help prevent the "exploding gradients" problem.
            clip_grad_norm_(model.parameters(), clip_value)
            
            # Update parameters and take a step using the computed gradient.
            optimizer.step()
            
            # update the lr
            scheduler.step()

            # Progress update every freq batches.
            if step % freq == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. loss: {}'.format(step, len(train_dataloader), elapsed, loss.item()))


        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()
        
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_score = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in dev_dataloader:
            batch = {k:v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = loss_function(outputs.logits.squeeze().float(), batch['labels'].float())
#             loss = outputs.loss
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = batch['labels'].to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_score += r2_score(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_score = total_eval_score / len(dev_dataloader)
        print("  MSE: {0:.2f}".format(avg_val_score))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(dev_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_score,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        
        es.check_early_stopping(avg_val_loss)
        if es.stop:
            print('Early Stopped.')
            break
        
        if not os.path.isdir(config['output_dir']):
            os.mkdir(config['output_dir'])
        if best_score < avg_val_score:
            best_score = avg_val_score
            check_point = {
                'model': model.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict()
            }

            torch.save(check_point, os.path.join(config['output_dir'],'best_model_ckpt.pt'))
            print(f"model save - {os.path.join(config['output_dir'],'best_model_ckpt.pt')} ")
        else:
            torch.save(check_point, os.path.join(config['output_dir'],'model_ckpt.pt'))
            print(f"model save - {os.path.join(config['output_dir'],'model_ckpt.pt')} ")
            

    print("\nTraining complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))



set_seed(CONFIG['seed'])

tokenizer = AutoTokenizer.from_pretrained(CONFIG['pretrained_model'])
train_dataset = RegressionDataset(tokenizer=tokenizer, file_path=CONFIG['train_file'], dir_path=CONFIG['output_dir'], mode='train')
train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], shuffle=True)
dev_dataset = RegressionDataset(tokenizer=tokenizer, file_path=CONFIG['dev_file'], dir_path=CONFIG['output_dir'], mode='dev')
dev_dataloader = DataLoader(dev_dataset, batch_size=CONFIG['dev_batch_size'], shuffle=False)

# model = LMRegressor(CONFIG['pretrained_model'], CONFIG['num_classes'])
model = AutoModelForSequenceClassification.from_pretrained(CONFIG['pretrained_model'], num_labels=CONFIG['num_classes'])
# model = BertForSequenceClassification.from_pretrained(CONFIG['pretrained_model'], num_labels=CONFIG['num_classes'])

if torch.cuda.device_count() > 1:
    print('\ndata parallel..')
    print(f'device : {CONFIG["device"]}')
    print(f"device_ids : {CONFIG['device_ids']}")

    model = torch.nn.DataParallel(model, device_ids=CONFIG['device_ids'])
    model.to(CONFIG['device'])

# model.to(CONFIG['device'])

optimizer = AdamW(model.parameters(),
                  lr=CONFIG['lr'],
                  eps=1e-8,
                  correct_bias=False)

epochs = CONFIG['epochs']
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

loss_function = nn.MSELoss()
# loss_function = DataParallelCriterion(nn.MSELoss(), device_ids=CONFIG['device_ids'])


train(CONFIG, model, optimizer, scheduler, loss_function, CONFIG['epochs'],       
      train_dataloader, dev_dataloader, device=CONFIG['device'], clip_value=2, freq=1000, patience=4)

