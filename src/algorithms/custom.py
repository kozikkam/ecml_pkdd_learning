import numpy as np
class CustomAlgorithm:
    def __init__(self, features, alpha):
        self.features = features
        self.alpha = alpha

    def get_predicted_labels(self):
        features_square = np.square(self.features)
        features_reduced = np.sum(features_square, axis=1)
        labels = list([1 if x > self.alpha else 0 for x in features_reduced])
        return labels


