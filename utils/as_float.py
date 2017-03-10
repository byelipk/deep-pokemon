from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class AsFloat(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(np.float32)
