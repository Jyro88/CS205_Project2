import numpy as np

class Classifier:
    def __init__(self):
        self.training_data = None
        self.labels = None

    def train(self, training_data, labels):
        self.training_data = training_data
        self.labels = labels

    def test(self, test_instance):
        distances = np.linalg.norm(self.training_data - test_instance, axis=1)
        nearest_neighbor_idx = np.argmin(distances)
        return self.labels[nearest_neighbor_idx]
