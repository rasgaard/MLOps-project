from src.data.make_dataset import read_data
from src.features.build_features import encode_texts
from transformers import RobertaForSequenceClassification
import cProfile

def load_model():
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')

def profiling_test():
    train_texts, _, _, train_labels, _, _ = read_data()
    encoded_train = encode_texts(train_texts, train_labels)
    load_model()

if __name__ == '__main__':
    cProfile.run('profiling_test()', filename = './outputs/profiling.prof')
    
