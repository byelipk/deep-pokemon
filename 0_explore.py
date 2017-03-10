import pandas as pd

# What is the goal of this project?
#
# Using "Pokemon.csv" obtained on Kaggle.com, create a neural network that
# can predict, with greater than 75% accuracy, the "Type 1" class of the Pokemon.
#
#
# For info about the dataset see: https://www.kaggle.com/abcsds/pokemon


FILEPATH = "Pokemon.csv"
pokemon  = pd.read_csv(FILEPATH)

print(pokemon.info())
print()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 800 entries, 0 to 799
# Data columns (total 13 columns):
# #             800 non-null int64
# Name          800 non-null object
# Type 1        800 non-null object
# Type 2        414 non-null object
# Total         800 non-null int64
# HP            800 non-null int64
# Attack        800 non-null int64
# Defense       800 non-null int64
# Sp. Atk       800 non-null int64
# Sp. Def       800 non-null int64
# Speed         800 non-null int64
# Generation    800 non-null int64
# Legendary     800 non-null bool
# dtypes: bool(1), int64(9), object(3)
# memory usage: 75.9+ KB
# None
#
# What are yoru observations about the data?
#
# 1) It's a small dataset with 800 examples.
# 2) All integer values are 64-bits, which is probably too much precision
#    for what we want to accomplish.
# 3) Total is a sum of all other attributes. We'll want to discard this field.
# 4) Type 2 has almost 400 non-null values, so it's hard to see this as a
#    usefull attribute.

# Let's get rid of the columns we know we don't need
bad_labels = ['#', 'Name', 'Type 2', 'Total', 'Generation', 'Legendary']
pokemon = pokemon.drop(bad_labels, axis=1) # drop() creates a copy

print(pokemon["Type 1"].value_counts())
print()

# Water       112
# Normal       98
# Grass        70
# Bug          69
# Psychic      57
# Fire         52
# Rock         44
# Electric     44
# Ground       32
# Ghost        32
# Dragon       32
# Dark         31
# Poison       28
# Fighting     27
# Steel        27
# Ice          24
# Fairy        17
# Flying        4
# Name: Type 1, dtype: int64

from utils.check_distribution import *

print(check_distribution(pokemon, "Type 1"))
print()

# Water       0.14000
# Normal      0.12250
# Grass       0.08750
# Bug         0.08625
# Psychic     0.07125
# Fire        0.06500
# Rock        0.05500
# Electric    0.05500
# Ground      0.04000
# Dragon      0.04000
# Ghost       0.04000
# Dark        0.03875
# Poison      0.03500
# Fighting    0.03375
# Steel       0.03375
# Ice         0.03000
# Fairy       0.02125
# Flying      0.00500
# Name: Type 1, dtype: float64

print(pokemon.describe())
print()

# Rename our columns
pokemon.columns = [
    'type_1',
    'hit_points',
    'attack',
    'defense',
    'sp_attack',
    'sp_defense',
    'speed'
]

# Save our dataset.
pokemon.to_csv("pokemon_dataset.csv", index=False)

print(pokemon.info())
