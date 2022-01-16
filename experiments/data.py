import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import clean

class DataProcessor(Dataset):
    """
    get batch data and tokenizing
    """

    def __init__(self, df, tokenizer, is_eval=False) -> None:

        self.df = df
        self.tokenizer = tokenizer
        self.is_eval = is_eval

    def tokenize(self, text):
        # tokenize and to tensor
        try:
            embedding = self.tokenizer.encode_plus(
                                    text=text,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt"
                            )
            # print(text)
            # print(embedding['input_ids'])
            # print(embedding['token_type_ids'])
            # print(embedding['attention_mask'])

        except:
            print('text error')
            print(text)

        return embedding

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        if self.is_eval:
            inputs = self.tokenize(self.df[idx])
            return inputs
        else:
            inputs = self.tokenize(self.df['comment'].values[idx])
            labels = self.df['score'].values[idx]
            # print(inputs)
            # print(labels)
            return inputs, labels