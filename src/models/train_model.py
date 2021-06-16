import argparse
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import AdamW, RobertaForSequenceClassification

#import wandb
from src.data.make_dataset import read_data
from src.features.build_features import encode_texts

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# wandb.init()
start = time.time()


def train(epochs, lr):
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    model.to(device)
    model.train()

    #wandb.watch(model, log_freq=100)

    optim = AdamW(model.parameters(), lr=lr)
    train_texts, val_texts, train_labels, val_labels = read_data()
    encoded_train, encoded_val = encode_texts(train_texts, train_labels), encode_texts(val_texts, val_labels)
    train_loader = DataLoader(encoded_train, 4, shuffle=True, num_workers=4)
    val_loader = DataLoader(encoded_val, 4, shuffle=True, num_workers=4)

    lowest_val_loss = 1000
    for epoch in range(epochs):
        running_loss_train = 0
        running_loss_val = 0
        for batch_idx, batch in enumerate(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            running_loss_train += loss
            loss.backward()
            optim.step()
            print(f"\nTrain batch {batch_idx+1}/{len(train_loader)}", end='\r')
            # if batch_idx % 40 == 0:
            #wandb.log({"Loss": loss})

        

        with torch.no_grad():
            model.eval()
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                running_loss_val += loss
                print(f"\nValidation batch {batch_idx+1}/{len(val_loader)}", end='\r')

        # Save the model with the lowest validation loss
        if running_loss_val/len(val_loader) < lowest_val_loss:
            model.save_pretrained('models/Roberta-fakenews.pt')
            lowest_val_loss = running_loss_val/len(val_loader)

        print(f"\nEpoch {epoch+1}/{epochs}\
                \nTraining loss: {running_loss_train/len(train_loader)}\
                \nValidation loss: {running_loss_val/len(val_loader)}\n")

    # Execution time
    end = time.time()
    total_time = end - start
    print('Execution time: ', total_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=0.003)
    parser.add_argument("--epochs", default=3)
    args = parser.parse_args(sys.argv[1:])
    print(args)

    train(epochs=int(args.epochs), lr=float(args.lr))
