from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attributes].values
