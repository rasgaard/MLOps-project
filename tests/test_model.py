import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, RobertaForSequenceClassification

from src.data.make_dataset import read_data
from src.features.build_features import encode_texts


def test_model():
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    train_texts, _, _, train_labels, _, _ = read_data()
    encoded_train = encode_texts(train_texts, train_labels)
    batch_size = 3
    train_loader = DataLoader(encoded_train, batch_size=batch_size, shuffle=True)
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # Check that the output dimensions are correct
        assert outputs[1].shape == torch.Size([batch_size,2])
        
        # Check that the probabilities for the classes sum to 1
        probabilities = F.softmax(outputs[1], dim=1)
        classes_prob = probabilities.sum(dim=1)
        for prob in classes_prob:
            assert 1 == pytest.approx(prob.item(), 0.015)
        
        break

test_model()