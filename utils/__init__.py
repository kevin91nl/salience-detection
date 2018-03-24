def compute_jaccard_index(set1: set, set2: set):
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


def is_valid_esd_json(data: dict, is_train_document: bool = False):
    """Test whether a dictionary (parsed JSON) adheres to the input data format as specified in the README file.

    Parameters
    ----------
    data : dict
        The parsed JSON data (found by applying json.loads on the text contents of the input file).
    is_train_document : bool, optional
        Whether the document is a train or test document (default : False).

    Returns
    -------
    bool
        True if the data is a valid input file.

    Raises
    ------
    ValueError
        If the data is not a dictionary.
    ValueError
        If 'text' is not found in the data.
    ValueError
        if 'abstract' is not found in the train/test data.
    ValueError
        If 'entities' is not found in the train/test data.
    ValueError
        If data['entities'] is not a list in the train/test data.
    ValueError
        If there exists an entity without "salience" key.
    ValueError
        If there exists an entity without "entity" key.
    """
    if 'text' not in data:
        raise ValueError('The "text" field is not found in the data.')
    if is_train_document:
        if 'abstract' not in data:
            raise ValueError('The "abstract" field is not found in the train/test data.')
        if 'entities' not in data:
            raise ValueError('The "entities" field is not found in the train/test data.')
        if type(data['entities']) != list:
            raise ValueError('The "entities" field should be a list.')
        for entity in data['entities']:
            if 'salience' not in entity:
                raise ValueError('All entities should have a "salience" field.')
            if 'entity' not in entity:
                raise ValueError('All entities should have an "entity" field.')
    return True
