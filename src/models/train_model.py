import torch
from torch.utils.data import DataLoader
from transformers import AdamW, RobertaForSequenceClassification

import wandb
from src.data.make_dataset import get_encoded_data

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

wandb.init()
def train(epochs=3, lr=0.003):
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    model.to(device)
    model.train()

    wandb.watch(model, log_freq=100)

    optim = AdamW(model.parameters(), lr=lr)
    dataset = get_encoded_data(train=True)
    train_loader = DataLoader(dataset, 4, shuffle=True)

""".git/
    for epoch in range(epochs):
        running_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            running_loss += loss
            loss.backward()
            optim.step()
            print(f"Batch {batch_idx+1}/{len(train_loader)}", end='\r')
            if batch_idx % 40 == 0:
                wandb.log({"Loss": loss})

        print(f"\nEpoch {epoch+1}/{epochs}\tLoss: {running_loss/len(train_loader)}\n")
    # torch.save(model.state_dict(), 'models/checkpoint.pth')

    # Should be loadable with RobertaForSequenceClassification.from_pretrained('models/Roberta-fakenews.pt')
    model.save_pretrained('models/Roberta-fakenews.pt') 
"""

if __name__ == '__main__':
    train()