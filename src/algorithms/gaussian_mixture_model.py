from sklearn.mixture.gaussian_mixture import GaussianMixture

class GaussianMixtureModel:
    def __init__(self, k, features, covariance_type="full"):
        self.k = k
        self.features = features
        self.covariance_type = covariance_type

    def gaussian_mixture_model(self, features):
        return GaussianMixture(self.k, self.covariance_type).fit(features)

    def get_predicted_labels(self):
        return self.gaussian_mixture_model(self.features).predict(self.features)
