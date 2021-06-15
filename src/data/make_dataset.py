import pandas as pd
from sklearn.model_selection import train_test_split


def read_data():
    df = pd.read_csv(f'data/raw/train.csv')
    df = df.fillna('')
    texts, labels = list(df['text'].values), list(df['label'].values)

    # train og validation split
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, train_size=0.85, test_size = 0.15)

    return train_texts, val_texts, train_labels, val_labels

if __name__ == "__main__":
    pass
