import pandas as pd
from utils.read_target_label import *

# What is the goal of this project?
#
# Using "Pokemon.csv" obtained on Kaggle.com, create a neural network that
# can predict, with greater than 75% accuracy, the "Type 1" class of the Pokemon.
#
#
# For info about the dataset see: https://www.kaggle.com/abcsds/pokemon


FILEPATH = "Pokemon.csv"
pokemon  = pd.read_csv(FILEPATH)

target_label = read_target_label()

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


# List of columns
#
# #: ID for each pokemon
# Name: Name of each pokemon
# Type 1: Each pokemon has a type, this determines weakness/resistance to attacks
# Type 2: Some pokemon are dual type and have 2
# Total: sum of all stats that come after this, a general guide to how strong a pokemon is
# HP: hit points, or health, defines how much damage a pokemon can withstand before fainting
# Attack: the base modifier for normal attacks (eg. Scratch, Punch)
# Defense: the base damage resistance against normal attacks
# SP Atk: special attack, the base modifier for special attacks (e.g. fire blast, bubble beam)
# SP Def: the base damage resistance against special attacks
# Speed: determines which pokemon attacks first each round
#

# Let's get rid of the columns we know we don't need. The columns that
# are commented out are the ones we want to keep.
ignore_columns  = [
    '#',
    'Name',
    'Type 1',
    'Type 2',
    'Total',
    # 'HP',
    #'Attack',
    #'Defense',
    # 'Sp. Atk',
    # 'Sp. Def',
    # 'Speed',
    'Generation',
    # 'Legendary'
]
pokemon = pokemon.drop(ignore_columns, axis=1) # drop() creates a copy

print(pokemon[target_label].value_counts())
print()

from utils.check_distribution import *
print(check_distribution(pokemon, target_label))
print()

# print(pokemon.describe())
print(pokemon.corr())
print()

# Format column names
import re

def fmt_col(col):
    col = col.lower()
    col = col.replace(".", "")
    col = col.replace(" ", "_")
    return col

new_columns     = [fmt_col(attr) for attr in pokemon.columns]
pokemon.columns = new_columns

# Save our dataset.
pokemon.to_csv("pokemon_dataset.csv", index=False)

print(pokemon.info())
