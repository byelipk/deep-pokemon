from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class SingleLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, labels, y=None):
        return self

    def transform(self, labels):
        le     = LabelEncoder()
        labels = le.fit_transform(labels.ravel())
        return labels.reshape(-1, 1)
