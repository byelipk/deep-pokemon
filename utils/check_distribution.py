
def check_distribution(data, category):
    """
    Given a data frame and a target category, return the distribution
    of each category in the dataset.
    """
    return data[category].value_counts() / len(data)
