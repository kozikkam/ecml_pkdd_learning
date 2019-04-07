import numpy as np
import pandas as pd
from sklearn import metrics
from src.preprocessors import PFA
from src.algorithms import KMeansAlgorithm
from src.algorithms import GaussianMixtureModel
from src.algorithms import CustomAlgorithm
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

STATISTICS_COLUMNS = ['ALGORITHM','PARAMS', 'TP', 'TN', 'FP', 'FN', 'PRECISION', 'RECALL', 'ADJUSTED_RAND_SCORE',
                      'HOMOGENEITY_SCORE', 'COMPLETENESS_SCORE', 'V_MEASURE_SCORE', 'FOWLKES_MALLOWS_SCORE']

class DataAnalyzer:
    def __init__(self):
        self.features = None
        self.labels = None
        self.algorithms_results = []
        self.algorithms_statistics = pd.DataFrame(columns=STATISTICS_COLUMNS)

    def read_data(self, file_path="../../features-train.dat.npz"):
        data = np.load(file_path)
        self.features = data.f.arr_0
        self.labels = data.f.arr_1

    def get_features(self):
        return self.features

    def get_labels(self):
        return self.labels

    def add_algorithm_result(self, algorithm_name, predicted_results, params=""):
        self.algorithms_results.append((algorithm_name, predicted_results))
        algorithm_statistics = self.get_algorithm_statistics(algorithm_name, self.labels, predicted_results, params)
        self.algorithms_statistics = self.algorithms_statistics.append(pd.DataFrame([algorithm_statistics],
                                                                       columns=STATISTICS_COLUMNS),
                                                                       ignore_index=True)

    def get_algorithm_statistics(self, name, labels, predicted_labels, params=""):
        tn, fp, fn, tp = metrics.confusion_matrix(labels, predicted_labels, labels=[0, 1]).ravel()
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        adjusted_rand_score = metrics.adjusted_rand_score(labels, predicted_labels)
        homogeneity_score = metrics.homogeneity_score(labels, predicted_labels)
        completeness_score = metrics.completeness_score(labels, predicted_labels)
        v_measure_score = metrics.v_measure_score(labels, predicted_labels)
        fowlkes_mallows_score = metrics.fowlkes_mallows_score(labels, predicted_labels)
        return [name, params, tp, tn, fp, fn, precision, recall, adjusted_rand_score, homogeneity_score,
                completeness_score, v_measure_score, fowlkes_mallows_score]

    def show_statistics(self):
        statistics_to_show = self.algorithms_statistics.set_index('ALGORITHM')
        print(statistics_to_show.to_string())

    def reduce_features(self, n):
        pfa = PFA(n_features=n)
        pfa.fit(self.get_features())
        self.features = pfa.features_

    def compare_algorithms(self):
        return

    def save_statistics(self):
        return

    def get_reduced_nr_of_dimensions(self, final_dimensions):
        pca = PCA(n_components=final_dimensions)
        reduced_features = pca.fit_transform(self.features)
        return reduced_features

    def plot_2D_figure(self, predicted_labels, plot_name="SQL Injection Detection"):
        features = self.get_reduced_nr_of_dimensions(2)
        features_with_targets = np.concatenate((features, predicted_labels[:, None]), axis=1)
        targets = [0, 1]
        colors = ['g', 'r']
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(plot_name, fontsize=20)
        for target, color in zip(targets, colors):
            indices_to_keep = features_with_targets[:, 2] == target
            ax.scatter(features_with_targets[indices_to_keep][:, 0]
                       , features_with_targets[indices_to_keep][:, 1]
                       , c=color
                       , s=50)
        ax.legend(["No attack", "Attack"])
        ax.grid()
        plt.show()

dataAnalyzer = DataAnalyzer()
dataAnalyzer.read_data()
n_custers = 2


for i in ["full", "elkan"]:
    kMeans = KMeansAlgorithm(n_custers, dataAnalyzer.get_features(), i)
    kMeansPredictedValues = kMeans.get_predicted_labels()
    dataAnalyzer.add_algorithm_result("K-means", kMeansPredictedValues, "algorithm=" + i)

for i in ["full", "tied", "spherical"]:
    gmm = GaussianMixtureModel(n_custers, dataAnalyzer.get_features(), i)
    gmmPredictedValues = gmm.get_predicted_labels()
    dataAnalyzer.add_algorithm_result("GMM", gmmPredictedValues, "covariance_type=" + i)
    dataAnalyzer.plot_2D_figure(gmmPredictedValues, "GMM covariance_type=" + i)


for i in np.arange(0.0, 1.0, 0.1):
    customAlg = CustomAlgorithm(dataAnalyzer.get_features(), i)
    customAlgPredictedValues = customAlg.get_predicted_labels()
    dataAnalyzer.add_algorithm_result("CustomAlgorithm", customAlgPredictedValues, "alpha=" + str(i))

dataAnalyzer.show_statistics()
