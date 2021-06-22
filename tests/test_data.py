import numpy as np
import torch

from src.data.make_dataset import read_data
from src.features.build_features import encode_texts

# read raw data 
train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = read_data()

# test raw data 
def test_raw_data():
    assert len(set(train_labels)) == 2 # 2 classes 
    assert np.isnan(train_labels).sum() == 0 # no na's in labels 
    assert np.isnan(train_texts).sum() == 0 # no na's in texts
    assert len(train_texts) == len(train_labels) # no. text obs equals no. labels

test_raw_data()

# read encoded data 
encoded_train = encode_texts(train_texts, train_labels)
encoded_val = encode_texts(val_texts, val_labels)
encoded_test = encode_texts(test_texts, test_labels)

# test encoded data
def test_encoded_data():
    assert list(encoded_train.__getitem__(0).keys()) == ['input_ids', 'attention_mask', 'labels']
    assert list(encoded_val.__getitem__(0).keys()) == ['input_ids', 'attention_mask', 'labels']
    assert list(encoded_test.__getitem__(0).keys()) == ['input_ids', 'attention_mask', 'labels']
    assert encoded_train.__getitem__(0)['labels'].dtype is torch.int64
    assert encoded_train.__getitem__(0)['input_ids'].dtype is torch.int64
    assert encoded_val.__getitem__(0)['labels'].dtype is torch.int64
    assert encoded_val.__getitem__(0)['input_ids'].dtype is torch.int64
    assert encoded_test.__getitem__(0)['labels'].dtype is torch.int64
    assert encoded_test.__getitem__(0)['input_ids'].dtype is torch.int64

test_encoded_data()