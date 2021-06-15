import torch
from torch.utils.data import DataLoader
from transformers import AdamW, RobertaForSequenceClassification
import time

#import wandb
from src.data.make_dataset import get_encoded_data

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#wandb.init()
start = time.time()
def train(epochs=3, lr=0.003):
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    model.to(device)
    model.train()

    #wandb.watch(model, log_freq=100)

    optim = AdamW(model.parameters(), lr=lr)
    encoded_train, encoded_val = get_encoded_data(train=True)
    train_loader = DataLoader(encoded_train, 16, shuffle=True, num_workers = 4)
    val_loader = DataLoader(encoded_val, 16, shuffle=True, num_workers = 4)

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
            print(f"Batch {batch_idx+1}/{len(train_loader)}", end='\r')
            #if batch_idx % 40 == 0:
                #wandb.log({"Loss": loss})

        print(f"\nEpoch {epoch+1}/{epochs}\tLoss: {running_loss/len(train_loader)}\n")

        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            running_loss_val += loss
        
        # Save the model with the lowest validation loss 
        if running_loss_val < lowest_val_loss:
            model.save_pretrained('models/Roberta-fakenews.pt') 
            running_loss_val = lowest_val_loss


    # Execution time 
    end = time.time()
    total_time = end - start
    print('Execution time: ', total_time)

    
#if __name__ == '__main__':
#    train()