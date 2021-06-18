import argparse
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification

from src.data.make_dataset import read_data
from src.features.build_features import encode_texts

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def predict(model_name, logger):
    if logger is not None:
        if logger == 'azure':
            print("Monitoring on Azure")
            from azureml.core import Run
            run = Run.get_context()
            import joblib
        if logger == 'wandb':
            print("Monitoring with wandb")
        else:
            raise ValueError("Logger has to be either 'azure' or 'wandb'")

    model = RobertaForSequenceClassification.from_pretrained('roberta-base')

    if model_name != 'None':
        if logger == 'azure':
            model_path = f'models/' + model_name
            model = joblib.load(model_path)
            print('loading our model')
        else:
            model.load_state_dict(torch.load(model_name))

    model.to(device)
    model.eval()

    batch_size = 4
    _, _, test_texts, _, _, test_labels = read_data()
    encoded_test = encode_texts(test_texts, test_labels)
    test_loader = DataLoader(encoded_test, batch_size, shuffle=True)
    
    false = 0
    total = 0
    with torch.no_grad():
        preds = []
        True_labels = []
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
                True_labels.append(labels[i].item())
                print(probabilities[i][0])
                if probabilities[i][0] >= 0.5:
                    pred_labels.append(0)
                    preds.append(0)
                else:
                    pred_labels.append(1)
                    preds.append(1)

                false += abs(pred_labels[i] - labels[i])
            total += len(labels)
            running_accu = 1-(false/total)
            print("Running accuracy: ", running_accu)

    accuracy = 1-(false/total)
    print("Accuracy: ", accuracy)
    if logger is not None:
        if logger == 'azure':
            run.log("Accuray",accuracy.item())
            # log table with preds and labels
            for i in range(len(preds)):
                run.log_row("True_vs_pred", True_class = True_labels[i], Predicted_class=preds[i])
                # Confusion matrix
                cm = confusion_matrix(True_labels,preds)
                cm = {"schema_type": "confusion_matrix",
                    "schema_version": "1.0.0",
                    "data": {
                    "class_labels": ["0", "1"],
                    "matrix": [[int(y) for y in x] for x in cm]
                    }
                    }
                run.log_confusion_matrix('Confusion matrix', cm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prediction arguments")
    parser.add_argument("--m", default=None)
    parser.add_argument("--logger", default=None)
    args = parser.parse_args(sys.argv[1:])
    print(args)

    predict(model_name=str(args.m))
