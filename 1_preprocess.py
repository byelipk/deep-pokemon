import pandas as pd
import numpy as np

# Load our dataset
pokemon = pd.read_csv("pokemon_dataset.csv")

# Build a feature scaling pipeline.
#
# We've create two pipelines, one for the features and one for the label.
# Our features will be normalized and scaled appropriately. Any missing
# values will be set to that attribute's median value.
#
# Our label will be transformed from a text object to an integer in the
# range of [0, 17].
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

from utils.data_frame_selector import *
from utils.add_bias_term import *
from utils.single_label_encoder import *
from utils.as_float import *

# Each column except our output label can be considered
# a "numeric" column. We can preprocess these columns in
# the same pipeline.
num_cols = list(pokemon.drop(['type_1'], axis=1).columns)
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_cols)),
    ('imputer', Imputer(strategy="median")),
    ('std_scaler', StandardScaler()),
    ('bias_term', AddBiasTerm()),
    ('as_float', AsFloat()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(['type_1'])),
    ('encode', SingleLabelEncoder()),
])

full_pipeline = FeatureUnion(transformer_list = [
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])

# From now on our data is a numpy array.
pokemon_prep = full_pipeline.fit_transform(pokemon)

# ### Stratified Sampling ###
from sklearn.model_selection import StratifiedShuffleSplit
# #
# # Create test, validation, and training set.
# #
# #   - Use stratified sampling
# #
# #   Stratified Sampling
# #   ===================
# #
# #   Our dataset should be representative of the population we're trying to
# #   generalize about. The same holds true for our data splits. Each split we
# #   make should be representative of each class.
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(pokemon_prep, pokemon_prep[:, -1]):
    strat_train_set = pokemon_prep[train_idx, :]
    strat_test_set  = pokemon_prep[test_idx, :]

# Build the training and test set. The target labels are at index position -1.
training_set    = strat_train_set[:, 0:-1]
training_labels = strat_train_set[:, -1].astype(np.int32)

test_set    = strat_test_set[:, 0:-1]
test_labels = strat_test_set[:, -1].astype(np.int32)

# Save our data
np.save("training_set", training_set)
np.save("training_labels.npy", training_labels)

np.save("test_set.npy", test_set)
np.save("test_labels.npy", test_labels)

print("Training and test data saved successfully!")
