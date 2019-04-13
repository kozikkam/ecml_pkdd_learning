import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class KMeansAlgorithm:
    def __init__(self, k, features, algorithm="full"):
        self.k = k
        self.features = features
        self.algorithm = algorithm

    def k_means(self, features):
        return KMeans(self.k, algorithm=self.algorithm).fit(features)

    def get_predicted_labels(self):
        return self.k_means(self.features).labels_



