import argparse
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AdamW, RobertaForSequenceClassification

from src.data.make_dataset import read_data
from src.features.build_features import encode_texts

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def test_training_evaluation(epochs=1, lr=0.003):
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', output_hidden_states=True)
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=lr)
    train_texts, val_texts, _, train_labels, val_labels, _ = read_data()
    train_texts, val_texts, train_labels, val_labels = train_texts[:3], val_texts[:3], train_labels[:3], val_labels[:3]
    encoded_train, encoded_val = encode_texts(train_texts, train_labels), encode_texts(val_texts, val_labels)
    train_loader = DataLoader(encoded_train, 3, shuffle=True)
    val_loader = DataLoader(encoded_val, 3, shuffle=True)

    for epoch in range(epochs):
        weights = []
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

            w = []
            for name, param in model.named_parameters():
                w.append(param)
            weights.append(w)


            print(f"\nTrain batch {batch_idx+1}/{len(train_loader)}\n", end='\r')
    

        # Check that at least one weight changes after running each batch
        check_weights = []
        for i in range(len(weights)-1):
            for j in range(len(weights[i])):
                #print(weights[i][j].shape)
                #print(weights[i+1][j].shape)
                check_weights.append(weights[i][j] != weights[i+1][j])
            #print(check_weights[i])
            check_weights[i] = sum(sum(check_weights[i]))
            print(check_weights[i])
            #print(sum(sum(check_weights)))
            
            #assert sum(sum(sum(weights[i] != weights[i+1]))) > 0

        weights_eval = []
        with torch.no_grad():
            model.eval()
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]

                weights_eval.append(outputs[2][1])

                running_loss_val += loss
                print(f"\nValidation batch {batch_idx+1}/{len(train_loader)}\n", end='\r')
        
        # Check that the weights don't change in eval
        #for i in range(len(weights_eval)-1):
            #print(weights_eval[i])
            #print(weights_eval[i+1])
            #print(sum(sum(sum(weights_eval[i] == weights_eval[i+1]))))
            #assert sum(sum(sum(weights_eval[i] == weights_eval[i+1]))) > 0


        
    
test_training_evaluation()
