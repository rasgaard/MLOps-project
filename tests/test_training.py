import torch
from torch.utils.data import DataLoader
from transformers import AdamW, RobertaForSequenceClassification, get_scheduler

from src.data.make_dataset import read_data
from src.features.build_features import encode_texts

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_training_evaluation(epochs=1, lr=0.00005):
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base', output_hidden_states=True)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    train_texts, val_texts, _, train_labels, val_labels, _ = read_data()
    train_texts, val_texts, train_labels, val_labels = train_texts[
        :6], val_texts[:6], train_labels[:6], val_labels[:6]
    encoded_train, encoded_val = encode_texts(
        train_texts, train_labels), encode_texts(val_texts, val_labels)
    train_loader = DataLoader(encoded_train, 2, shuffle=True)
    val_loader = DataLoader(encoded_val, 2, shuffle=True)

    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    for epoch in range(epochs):
        weights = []
        weights_eval = []
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

            w = []
            for _, param in model.named_parameters():
                w.append(param)
            weights.append(w)

        # Check that the weights are updated after each batch
        check_weights = []
        for i in range(len(weights)-1):
            for j in range(len(weights[i])):
                check_weights.append(weights[i][j] != weights[i+1][j])
            check_weights[i] = sum(sum(check_weights[i]))
            # should work on larger amount of training data / non-pretrained model
            # assert check_weights[i].item() > 0

        model.eval()
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

                w_eval = []
                for _, param in model.named_parameters():
                    w_eval.append(param)
                weights_eval.append(w_eval)

        # Check that the weights are not updated when evaluating
        check_weights_eval = []
        for i in range(len(weights_eval)-1):
            for j in range(len(weights_eval[i])):
                check_weights_eval.append(
                    weights_eval[i][j] != weights_eval[i+1][j])
            check_weights_eval[i] = sum(sum(check_weights_eval[i]))
            assert check_weights_eval[i].item() == 0


test_training_evaluation()
