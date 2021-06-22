from src.models.train_model import train
from sklearn.decomposition import PCA
from src.data.make_dataset import read_data
from src.features.build_features import encode_texts
import matplotlib.pyplot as plt
import numpy as np


train_texts, _, _, train_labels, _, _ = read_data()

train_texts, train_labels = train_texts[:1000], train_labels[:1000]
encoded_texts = encode_texts(train_texts, train_labels)

data = np.array([encoding.ids for encoding in encoded_texts.encodings._encodings])

pca = PCA(n_components=2)

pca_fit = pca.fit(data)

transformed_data = pca_fit.transform(data)

plt.scatter(transformed_data[:,0], transformed_data[:,1], c=train_labels)
plt.show()