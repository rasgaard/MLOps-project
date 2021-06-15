from src.data.make_dataset import read_data, get_encoded_data
from src.features.build_features import encode_text
import numpy as np
import torch

# read raw data 
text, labels = read_data()

# test raw data 
def test_raw_data():
    assert len(set(labels)) == 2 # 2 classes 
    assert np.isnan(labels).sum() == 0 # no na's in labels 
    assert len(text) == len(labels) # no. text obs equals no. labels
test_raw_data()

# read encoded data 
encoded_train, encoded_val = get_encoded_data(train=True)

# test encoded data
def test_encoded_data():
    assert list(encoded_train.__getitem__(0).keys()) == ['input_ids', 'attention_mask', 'labels']
    assert list(encoded_val.__getitem__(0).keys()) == ['input_ids', 'attention_mask', 'labels']
    assert encoded_train.__getitem__(0)['labels'].dtype is torch.int64
    assert encoded_train.__getitem__(0)['input_ids'].dtype is torch.int64
    assert encoded_val.__getitem__(0)['labels'].dtype is torch.int64
    assert encoded_val.__getitem__(0)['input_ids'].dtype is torch.int64
test_encoded_data()
