# -*- coding: utf-8 -*-
import pandas as pd
import datasets as ds

df = pd.read_csv('../../data/raw/train.csv')
df = df.fillna('')
df = df.rename(columns={'label': 'labels'})
df['text'] = df['title'] + '\n\n' + df['text']
df = df.drop(columns=['id', 'title'])

dataset = ds.Dataset.from_pandas(df)
dataset.features['labels'] = ds.ClassLabel(num_classes=2, names=['unreliable', 'reliable'])