def compute_jaccard_index(set1: set, set2: set) -> float:
    """Given two sets, compute the Jaccard index.

    Parameters
    ----------
    set1 : set
        The first set.
    set2 : set
        The second set.

    Returns
    -------
    float
        The Jaccard index.
    """
    if len(set1) + len(set2) == 0:
        raise ValueError('There should at least be one element in set1 and set2.')
    return len(set1.intersection(set2)) / float(len((set1.union(set2))))
