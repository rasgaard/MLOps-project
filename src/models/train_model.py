import argparse
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, RobertaForSequenceClassification, get_scheduler

from src.data.make_dataset import read_data, seed_everything
from src.features.build_features import encode_texts

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(epochs, learning_rate, logger, batch_size, num_workers, seed):

    if logger is not None:
        if logger == 'azure':
            print("Monitoring on Azure")
            from azureml.core import Run
            run = Run.get_context()
        elif logger == 'wandb':
            print("Monitoring with wandb")
            import wandb
            wandb.init(config={"epochs": epochs, 
                               "lr": learning_rate, 
                               "batch_size": batch_size, 
                               "num_workers":num_workers,
                               "seed": args.seed})
        else:
            raise ValueError("Logger has to be either 'azure' or 'wandb'")

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    train_texts, val_texts, _, train_labels, val_labels, _ = read_data()

    encoded_train = encode_texts(train_texts, train_labels)
    encoded_val = encode_texts(val_texts, val_labels)

    train_loader = DataLoader(encoded_train, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(encoded_val, batch_size, shuffle=True, num_workers=num_workers)

    num_training_steps = epochs * len(train_loader)
    train_progress_bar = tqdm(range(num_training_steps))
    val_progress_bar = tqdm(range(epochs * len(val_loader)))
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    for epoch in range(epochs):
        running_train_loss = 0

        model.train()
        for batch in train_loader:
            # puts tensors in batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            running_train_loss += loss.item()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_progress_bar.update(1)

            if logger is not None:
                if logger == 'azure':
                    run.log('Training loss (each batch)', loss.item())
                if logger == 'wandb':
                    wandb.log({"Training loss (each batch)": loss.item()})

        running_val_loss = 0
        running_val_acc = 0

        model.eval()
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            accuracy = torch.mean((predictions == batch['labels']).float())
            running_val_loss += outputs.loss.item()
            running_val_acc += accuracy.item()
            val_progress_bar.update(1)

            if logger is not None:
                if logger == 'azure':
                    run.log('Validation loss (each batch)',
                            outputs.loss.item())
                    run.log("Validation accuracy (each batch)", accuracy.item())
                if logger == 'wandb':
                    wandb.log({"Validation loss (each batch)": outputs.loss.item(),
                               "Validation accuracy (each batch)": accuracy.item()})

        print(f"\nEpoch {epoch+1}/{epochs}\
                \nTraining loss: {running_train_loss/len(train_loader)}\
                \nValidation loss: {running_val_loss/len(val_loader)}\
                \nValidation accuracy: {running_val_acc/len(val_loader)}")

        if logger is not None:
            if logger == 'azure':
                run.log("Trainig loss", running_train_loss/len(train_loader))
                run.log("Validation loss", running_val_loss/len(val_loader))
                run.log("Validation accuracy", running_val_acc/len(val_loader))
            if logger == 'wandb':
                wandb.log({"Training loss": running_train_loss/len(train_loader),
                           "Validation loss": running_val_loss/len(val_loader),
                           "Validation accuracy": running_val_acc/len(val_loader)})

    model.save_pretrained(
        f"models/roberta_fakenews_lr={learning_rate}_epochs={epochs}_batchsize={batch_size}_numworkers={num_workers}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--n_workers", default=2, type=int)
    parser.add_argument("--logger", default=None)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args(sys.argv[1:])
    print(args)

    seed_everything(args.seed)

    train(epochs=args.epochs, learning_rate=args.lr, logger=args.logger, batch_size=args.batch_size, num_workers=args.n_workers, seed=args.seed)
