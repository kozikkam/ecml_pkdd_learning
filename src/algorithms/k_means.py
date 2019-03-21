import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class KMeansAlgorithm:
    def __init__(self, k):
        self.k = k
        self.features = None
        self.labels = None

    def read_data(self, file_path="../../features-train.dat.npz"):
        data = np.load(file_path)
        self.features = data.f.arr_0
        self.labels = data.f.arr_1

    def k_means(self, features):
        return KMeans(self.k).fit(features)

    def get_features(self):
        return self.features

    def get_labels(self):
        return self.labels

    def get_reduced_nr_of_dimensions(self, final_dimensions):
        pca = PCA(n_components=final_dimensions)
        reduced_features = pca.fit_transform(self.features)
        return reduced_features

    def plot_2D_figure(self):
        features = self.get_reduced_nr_of_dimensions(2)
        kmodel = self.k_means(features)
        features_with_targets = np.concatenate((features, kmodel.labels_[:, None]), axis=1)
        targets = [0, 1]
        colors = ['g', 'r']
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Dimension 1', fontsize=15)
        ax.set_ylabel('Dimension 2', fontsize=15)
        ax.set_title('SQL Injection Detection', fontsize=20)
        for target, color in zip(targets, colors):
            indices_to_keep = features_with_targets[:, 2] == target
            ax.scatter(features_with_targets[indices_to_keep][:, 0]
                       , features_with_targets[indices_to_keep][:, 1]
                       , c=color
                       , s=50)
        ax.legend(["No attack", "Attack"])
        ax.grid()
        plt.show()


km = KMeansAlgorithm(2)
km.read_data()
km.plot_2D_figure()

