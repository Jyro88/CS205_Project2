import numpy as np

def load_data(filename):
    data = np.loadtxt(filename)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    features = (features - features.mean(axis=0)) / features.std(axis=0, ddof=1)
    return features, labels
