import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn_bert_featurizer import BertTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, FeatureUnion
import pickle

from utils import getData 

import time

CONFIG = dict(
    seed = 12345,
    pretrained_model = "vinai/bertweet-base",
    output_dir = '../models/tfidf-char-bertemb-ridge-v1',
    train_file = '4th/v1/train.csv',
    #train_file = '4th/v0/train_mini.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    device_ids = [0,1]    
)

tokenizer = AutoTokenizer.from_pretrained(CONFIG['pretrained_model'])

model = AutoModel.from_pretrained(CONFIG['pretrained_model'])
if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=CONFIG['device_ids'])
        model.to(CONFIG['device'])

# def identity_emb(text):
#     tokenized_txt = torch.tensor(tokenizer.encode_plus(text=text,
#                                       padding='max_length',
#                                       truncation=True
#                                       )['input_ids']).to(CONFIG['device'])
#     features = model(tokenized_txt.unsqueeze(0))
#     return features


def main():

    # 1. load dataset
    df = getData(data_path=CONFIG['train_file'])
    df_val = getData(data_path="4th/validation_cleaned.csv")

    val_preds_arr1 = np.zeros((df_val.shape[0], 1))
    val_preds_arr2 = np.zeros((df_val.shape[0], 1))
    # 2. model

    bert_featurizer = BertTransformer(tokenizer, model)

    # feature using tfidf
    features = FeatureUnion([
            ("vect1", TfidfVectorizer(analyzer = 'char_wb', ngram_range = (3,5))),
            ("vect2", TfidfVectorizer(analyzer = 'word', stop_words="english")),
            ("vect3", bert_featurizer),
        ])

    regressor = Ridge(alpha=10)
    
    # pipeline using ridge regression
    pipeline = Pipeline(
            [
                ("features", features),
                ("regressor", regressor),
            ],
            verbose=True
        )

    # Train the pipeline
    start = time.time()
    print('\n model training \n')
    pipeline.fit(df['comment'], df['score'])
    print(f'training time : {time.time()-start}\n')

    if not os.path.isdir(CONFIG['output_dir']):
        os.mkdir(CONFIG['output_dir'])

    # save model
    with open(os.path.join(CONFIG['output_dir'], 'model.pkl'),'wb') as f:
        pickle.dump(pipeline,f)

    # 3. Validation

    print("\npredict validation data ")
    val_preds_arr1[:,0] = pipeline.predict(df_val['less_toxic'])
    val_preds_arr2[:,0] = pipeline.predict(df_val['more_toxic'])

    p1 = val_preds_arr1.mean(axis=1)
    p2 = val_preds_arr2.mean(axis=1)

    acc = np.round((p1 < p2).mean() * 100,2)

    print('=============================')
    print(CONFIG['output_dir'])
    print(f'Validation Accuracy is { np.round((p1 < p2).mean() * 100,2)}\n')
    print('=============================')

    with open(os.path.join(CONFIG['output_dir'], 'val_acc.txt'),'w') as f:
        f.write(f'{CONFIG["output_dir"]}\nvalidation accuracy : {acc}')

if __name__=='__main__':
    main()
