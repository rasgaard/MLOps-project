import torch
from transformers import RobertaTokenizerFast
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        super().__init__()
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def encode_text(texts, labels):
    tokenizer = RobertaTokenizerFast.from_pretrained(
        'roberta-base', max_length=512)
    encodings = tokenizer(texts, truncation=True, padding=True)
    return FakeNewsDataset(encodings, labels)
