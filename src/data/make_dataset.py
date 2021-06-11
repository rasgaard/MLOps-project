# -*- coding: utf-8 -*-
import pickle

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
    encoded_data = encode_text(texts, labels)
    # with open("data/processed/dataset.pkl", 'wb') as f:
    #     pickle.dump(encoded_data, f)
    return encoded_data

if __name__ == "__main__":
    print(get_encoded_data())
