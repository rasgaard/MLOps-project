# -*- coding: utf-8 -*-
import pickle
import datasets as ds 
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
from src.features.build_features import encode_text
# Hej

def read_data(train=True):
    file = 'train' if train else 'test'
    df = pd.read_csv(f'data/raw/{file}.csv')
    df = df.fillna('')
    df = df.rename(columns={'label': 'labels'})
    df['text'] = df['title'] + '\n\n' + df['text']
    df = df.drop(columns=['id', 'title', 'author'])

    return list(df['text'].values), list(df['labels'].values)

def get_encoded_data(train=True):
    texts, labels = read_data(train)
    # train og validation split
    train_text, val_text, train_labels, val_labels = train_test_split(texts, labels, train_size=0.85, test_size = 0.15)
    encoded_train = encode_text(train_text, train_labels)
    encoded_val = encode_text(val_text, val_labels)
    return encoded_train, encoded_val 

if __name__ == "__main__":
    print(get_encoded_data())
