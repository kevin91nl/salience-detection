def compute_jaccard_index(set1: set, set2: set):
    """Given two sets, compute the Jaccard index.

    :param set1: The first set.
    :param set2: The second set.
    :return: The Jaccard index.
    """
    if len(set1) + len(set2) == 0:
        raise ValueError('There should at least be one element in set1 and set2.')
    return len(set1.intersection(set2)) / float(len((set1.union(set2))))
