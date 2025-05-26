import numpy as np
from nearest_neighbor_classifier import Classifier

class Validator:
    def __init__(self, classifier, data, labels):
        self.classifier = classifier
        self.data = data
        self.labels = labels

    def evaluate(self, feature_subset):
        correct_predictions = 0
        num_instances = len(self.labels)
        for i in range(num_instances):
            training_data = np.delete(self.data, i, axis=0)
            training_labels = np.delete(self.labels, i, axis=0)
            test_instance = self.data[i, feature_subset]
            true_label = self.labels[i]

            self.classifier.train(training_data[:, feature_subset], training_labels)
            predicted_label = self.classifier.test(test_instance)

            if predicted_label == true_label:
                correct_predictions += 1

        accuracy = correct_predictions / num_instances
        return accuracy

def evaluate_subset(subset, data, labels):
    classifier = Classifier()
    validator = Validator(classifier, data, labels)
    accuracy = validator.evaluate(list(subset))
    return accuracy
