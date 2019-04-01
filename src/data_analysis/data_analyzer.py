import numpy as np
from sklearn import metrics
from src.algorithms import KMeansAlgorithm


class DataAnalyzer:
    def __init__(self):
        self.features = None
        self.labels = None
        self.algorithms_results = []

    def read_data(self, file_path="../../features-train.dat.npz"):
        data = np.load(file_path)
        self.features = data.f.arr_0
        self.labels = data.f.arr_1

    def get_features(self):
        return self.features

    def get_labels(self):
        return self.labels

    def add_algorithm_result(self, algorithm_name, predicted_results):
        self.algorithms_results.append((algorithm_name, predicted_results))

    def show_statistics(self):
        labels = self.get_labels()
        if len(self.algorithms_results) > 0:
            print("Algorithms results statistics: ")
            for result in self.algorithms_results:
                algorithm_name = result[0]
                predicted_labels = result[1]
                print("--- ", algorithm_name, " ---")
                print("Confusion matrix:")
                tn, fp, fn, tp = metrics.confusion_matrix(labels, predicted_labels, labels=[0, 1]).ravel()
                print("TN:", tn, "FP:", fp)
                print("FN:", fn, "TP:", tp)
                print("Precision:", tp/(tp + fp))
                print("Recall:", tp/(tp + fn))
                print("Adjusted rand score:", metrics.adjusted_rand_score(labels, predicted_labels))
                print("Homogenity score:", metrics.homogeneity_score(labels, predicted_labels))
                print("Completeness score:", metrics.completeness_score(labels, predicted_labels))
                print("V measure score:", metrics.v_measure_score(labels, predicted_labels))
                print("Fowlkes mallows score:", metrics.fowlkes_mallows_score(labels, predicted_labels))
        else:
            print("No data results to analyse")

    def compare_algorithms(self):
        return

    def save_statistics(self):
        return


dataAnalyzer = DataAnalyzer()
dataAnalyzer.read_data()

kMeans = KMeansAlgorithm(2, dataAnalyzer.features)
kMeansPredictedValues = kMeans.get_predicted_labels()

dataAnalyzer.add_algorithm_result("K-means", kMeansPredictedValues)
dataAnalyzer.show_statistics()