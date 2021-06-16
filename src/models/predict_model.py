import torch

import argparse
import sys
import torch.nn.functional as F

from src.data.make_dataset import read_data
from src.features.build_features import encode_texts

from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def predict(model_name):
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    
    if model_name != 'None':
        model.load_state_dict(torch.load(model_name))

    model.to(device)    
    model.eval()
    
    batch_size = 4
    _, val_texts, _, val_labels = read_data()
    encoded_val = encode_texts(val_texts, val_labels)
    val_loader = DataLoader(encoded_val, batch_size, shuffle=True)
    
    false = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            #print("Outputs: ", outputs)

            probabilities = F.softmax(outputs[1], dim=1)
            pred_labels = []
            print("Probs: ", probabilities)
            print("Labels: ", labels)
        
            for i in range(len(labels)):
                if probabilities[i][0] >= 0.5:
                    pred_labels.append(0)
                else:
                    pred_labels.append(1)
            
                false += abs(pred_labels[i] - labels[i])
            total += len(labels)
            running_accu = 1-(false/total)
            print("Running accuracy: ", running_accu)

    accuracy = 1-(false/total)
    print("Accuracy: ", accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--m", default=None)
    args = parser.parse_args(sys.argv[1:])
    print(args)

    predict(model_name=str(args.m))