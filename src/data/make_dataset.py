import pandas as pd
from sklearn.model_selection import train_test_split


def read_data():
    df = pd.read_csv(f'data/raw/train.csv')
    df = df.fillna('')
    texts, labels = list(df['text'].values), list(df['label'].values)

    # train, validation and test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, train_size=0.70, test_size=0.30, shuffle=False)
    test_texts, val_texts, test_labels, val_labels = train_test_split(test_texts, test_labels, train_size=0.5, test_size=0.5, shuffle=False)

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


if __name__ == "__main__":
    pass
