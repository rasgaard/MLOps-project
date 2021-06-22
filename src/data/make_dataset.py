import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def read_data():
    df = pd.read_csv(f'data/raw/train.csv')
    df = df.fillna('')
    texts, labels = list(df['text'].values), list(df['label'].values)

    # train, validation and test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, train_size=0.70, test_size=0.30, shuffle=True)
    test_texts, val_texts, test_labels, val_labels = train_test_split(test_texts, test_labels, train_size=0.5, test_size=0.5, shuffle=True)

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


# https://datascience.stackexchange.com/questions/66345/why-ml-model-produces-different-results-despite-random-state-defined-and-how-to
def seed_everything(seed=42):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    pass
