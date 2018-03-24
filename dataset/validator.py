def is_valid_esd_json(data: dict, is_train_document: bool = False) -> bool:
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
        If 'entities' is not a list in the train/test data.
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
