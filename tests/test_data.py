from make_dataset import read_data
import pandas as pd
import datasets as ds
import pickle
import torch

dataset = read_data()
#print(dataset.column_names)

# test of labels 
def test_labels():
    #print(len(dataset.features['labels'].names))
    assert len(dataset.features['labels'].names) == 2
    assert dataset.features['labels'].names[0] == 'unreliable'
    assert dataset.features['labels'].names[1] == 'reliable'
test_labels()

# test names and lengths of columns 
def test_columns():
    assert dataset.column_names == ['text', 'labels']
    assert len(dataset['text']) == len(dataset['labels'])
test_columns()

# test no missing values
def test_na():
    dataset_pd = pd.DataFrame(dataset)
    #print(dataset_pd.isna().sum().sum())
    assert dataset_pd.isna().sum().sum() == 0
test_na()

#train_data, val_data = preprocess()
with open('data/processed/train_tokenized.pickle', 'rb') as f:
        train_data = pickle.load(f)

with open('data/processed/val_tokenized.pickle', 'rb') as f:
        val_data = pickle.load(f)

def test_tokenizer():
    assert train_data.column_names == ['attention_mask', 'author', 'input_ids', 'labels', 'text']
    assert val_data.column_names == ['attention_mask', 'author', 'input_ids', 'labels', 'text']
    assert train_data['input_ids'].dtype is torch.int64
    assert train_data['labels'].dtype is torch.int64
    assert val_data['input_ids'].dtype is torch.int64
    assert val_data['labels'].dtype is torch.int64
test_tokenizer()