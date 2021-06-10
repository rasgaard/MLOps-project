from transformers import RobertaForSequenceClassification, AdamW
from src.data.make_dataset import get_encoded_data
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train():
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=0.003)
    dataset = get_encoded_data()
    train_loader = DataLoader(dataset, 4, shuffle=True)
    epochs = 3

    for epoch in range(epochs):
        running_loss = 0
        for step, batch in enumerate(train_loader):
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
            print(f"Batch {step}/{len(train_loader)}", end='\r')
        print(f"Epoch {epoch}/{epochs}\tLoss: {running_loss/len(batch)}")


if __name__ == '__main__':
    train()