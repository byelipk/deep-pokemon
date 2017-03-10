from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class AsInt(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, labels, **kwargs):
        return self

    def transform(self, labels):
        return labels.astype(np.int32)
