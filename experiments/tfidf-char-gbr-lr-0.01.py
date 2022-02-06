"""
features : tfidf char_wb
regressor : gradient boosting regressor (lr 0.01)
data: v1
"""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import set_config
import pickle
import json
import time
from utils import getData, set_seed, format_time

CONFIG = dict(
    seed = 12345,
    pretrained_model = 'bert-base-uncased',
    output_dir = '../models/tfidf-char-gbr-lr-0.01-v1',
    train_file = '4th/v1/train.csv',
    dev_file = '4th/v1/dev.csv',
    # output_dir = '../models/tfidf-char-svr-v1-test',
    # train_file = '../data/4th/v0/tr.csv',
    # dev_file = '../data/4th/v0/dv.csv',
    lr = 0.01,
)


def main():
    set_seed(CONFIG['seed'])
    # load data
    # train data
    df = getData(data_path=CONFIG['train_file'])
    # Validation data 
    df_val = getData(data_path="4th/validation_cleaned.csv")
    # Test data
    # df_sub = getData(data_path="4th/comments_to_score.csv")    


    val_preds_arr1 = np.zeros((df_val.shape[0], 1))
    val_preds_arr2 = np.zeros((df_val.shape[0], 1))
    # test_preds_arr = np.zeros((df_sub.shape[0], 1))

    # Measure how long the training epoch takes.
    t0 = time.time()

    # feature using tfidf
    features = FeatureUnion([
            ("vect1", TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5))),
        ])

    # pipeline using ridge regression
    pipeline = Pipeline(
            [
                ("features", features),
                ("regression", gbr(random_state=CONFIG['seed'], learning_rate=CONFIG['lr'], verbose=10)),
            ],
            verbose=True
        )

    set_config(display="diagram")

    # set model
    # Train the pipeline
    pipeline.fit(df['comment'], df['score'])

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("*** Training took: {:}".format(training_time))

    # save model
    if not os.path.isdir(CONFIG['output_dir']):
        os.mkdir(CONFIG['output_dir'])

    with open(os.path.join(CONFIG['output_dir'],'model.pkl'),'wb') as f:
        pickle.dump(pipeline,f)

    # config 파일 확인해서 없으면 저장
    if not os.path.isfile(os.path.join(CONFIG['output_dir'],'config.json')):
        with open(os.path.join(CONFIG['output_dir'],'config.json'), 'w') as writer:
            json.dump(CONFIG, writer)
    print(f'*** {CONFIG["ouput_dir"]} saved.')

    # validation
    # validate and test
    print("\npredict validation data ")
    val_preds_arr1[:,0] = pipeline.predict(df_val['less_toxic'])
    val_preds_arr2[:,0] = pipeline.predict(df_val['more_toxic'])

    # print("\npredict test data ")
    # test_preds_arr[:,0] = pipeline.predict(df_sub['text'])

    p1 = val_preds_arr1.mean(axis=1)
    p2 = val_preds_arr2.mean(axis=1)
    acc = np.round((p1 < p2).mean() * 100,2)
    print(f'Validation Accuracy is {acc }')
    with open(os.path.join(CONFIG['output_dir'], 'validation_result.txt'), 'w') as writer:
        result_text = f"Model: {CONFIG['output_dir']}\nAccuracy: {acc}"
        writer.write(result_text)

    # test
    # df_sub['score'] = test_preds_arr.mean(axis=1)
    
    # save submission
    # df_sub[['comment_id', 'score']].to_csv("submission.csv", index=False)


if __name__=='__main__':
    main()