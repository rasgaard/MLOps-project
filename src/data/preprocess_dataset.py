
import datasets as ds
from transformers import RobertaTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

def preprocess():
    with open('data/processed/dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)
        
    train_data, val_data = dataset.train_test_split(test_size=0.9).values()

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length=512)

    # define a function that will tokenize the model, and will return the relevant inputs for the model
    def tokenization(batched_text):
        return tokenizer(batched_text['text'], padding=True, truncation=True)

    train_data = train_data.map(tokenization, batched=True, batch_size=len(train_data))
    val_data = val_data.map(tokenization, batched=True, batch_size=len(val_data))
    #print(train_data.column_names)
    #print(type(train_data['input_ids'][0][0]))
    #print(train_data['text'][0:10])
    #print(train_data['input_ids'][0:10]) # input_ids is the text after tokenizer 

    train_data.set_format('torch', columns=['input_ids', 'labels'])
    val_data.set_format('torch', columns=['input_ids', 'labels'])
    #train_data.set_format('torch')
    #test_data.set_format('torch')

    with open('data/processed/train_tokenized.pickle', 'wb') as f:
        pickle.dump(train_data, f)

    with open('data/processed/val_tokenized.pickle', 'wb') as f:
        pickle.dump(val_data, f)

    return train_data, val_data

#preprocess()
