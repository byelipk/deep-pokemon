from sklearn.base import BaseEstimator, TransformerMixin

class AddBiasTerm(BaseEstimator, TransformerMixin):
    def __init__(self, enabled=False):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
