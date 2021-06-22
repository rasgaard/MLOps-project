import argparse
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification

from src.data.make_dataset import read_data, seed_everything
from src.features.build_features import encode_texts

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def predict(model_name, seed):
    model = RobertaForSequenceClassification.from_pretrained(f"models/{model_name}")

    model.to(device)
    model.eval()

    batch_size = 4
    _, _, test_texts, _, _, test_labels = read_data()
    encoded_test = encode_texts(test_texts, test_labels)
    test_loader = DataLoader(encoded_test, batch_size, shuffle=True)
    
    false = 0
    total = 0
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            # Logits which change every time the output is generated
            print("Logits: ", outputs.logits)

            probabilities = F.softmax(outputs.logits, dim=-1)
            pred_labels = []
            print("Probs: ", probabilities)
            print("Labels: ", labels)

            for i in range(len(labels)):
                print(probabilities[i][0])
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
    parser = argparse.ArgumentParser(description="Prediction arguments")
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args(sys.argv[1:])
    print(args)
    #tilføj seed
    seed_everything(args.seed)

    predict(model_name=args.model_name, seed=args.seed)
