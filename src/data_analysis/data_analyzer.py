import numpy as np
import pandas as pd
from sklearn import metrics
from src.preprocessors import PFA
from src.algorithms import KMeansAlgorithm
from src.algorithms import GaussianMixtureModel

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


dataAnalyzer = DataAnalyzer()
dataAnalyzer.read_data()

kMeans = KMeansAlgorithm(2, dataAnalyzer.get_features())
kMeansPredictedValues = kMeans.get_predicted_labels()
dataAnalyzer.add_algorithm_result("K-means", kMeansPredictedValues, "algorithm=full")
kMeans = KMeansAlgorithm(2, dataAnalyzer.get_features(), "elkan")
kMeansPredictedValues = kMeans.get_predicted_labels()
dataAnalyzer.add_algorithm_result("K-means", kMeansPredictedValues, "algorithm=elkan")

gmm = GaussianMixtureModel(2, dataAnalyzer.get_features())
gmmPredictedValues = gmm.get_predicted_labels()
dataAnalyzer.add_algorithm_result("GMM", gmmPredictedValues, "covariance_type=full")
gmm = GaussianMixtureModel(2, dataAnalyzer.get_features(), "tied")
gmmPredictedValues = gmm.get_predicted_labels()
dataAnalyzer.add_algorithm_result("GMM", gmmPredictedValues, "covariance_type=tied")
gmm = GaussianMixtureModel(2, dataAnalyzer.get_features(), "spherical")
gmmPredictedValues = gmm.get_predicted_labels()
dataAnalyzer.add_algorithm_result("GMM", gmmPredictedValues, "covariance_type=spherical")

dataAnalyzer.show_statistics()

