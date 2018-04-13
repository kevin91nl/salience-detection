import numpy as np


def convert_rsd_batch(batch, device=None):
    """Convert a batch of RSD data to the format the RSD model works with.

    Parameters
    ----------
    batch : list
        List of dictionaries.
    device : str, optional
        The device to use (default: None).

    Returns
    -------
    Dictionary such that each field points to a batched list.
    """
    if device is not None:
        raise NotImplementedError()
    return {
        'xs': [item['features'] for item in batch],
        'ts': np.asarray([int(item['is_relevant']) for item in batch], dtype='i')
    }


def convert_entmem_batch(batch, device=None):
    """Convert a batch of EntMem data to the format the EntMem model works with.

    Parameters
    ----------
    batch : list
        List of examples.
    device : str, optional
        The device to use (default: None).

    Returns
    -------
    Dictionary such that each field points to a batched list.
    """
    if device is not None:
        raise NotImplementedError()
    result = {
        'xs_feat': [],
        'idx_postags': [],
        'idx_words': [],
        'txt_words': [],
        'txt_postags': [],
        'is_salient': []
    }
    for item in batch:
        result['xs_feat'].append(
            np.vstack([
                np.array(item['features'])[:, :, 0],
                np.matrix(item['query_entity_features']).T,
                np.matrix(item['context_entities_features']).T,
            ])
        )
        result['idx_postags'].append(np.array(item['postag_ids']))
        result['idx_words'].append(np.array(item['word_ids']))
        result['txt_words'].append(item['words'])
        result['txt_postags'].append(item['postags'])
        result['is_salient'].append(item['is_salient'])
    result['is_salient'] = np.array(result['is_salient'])
    return result
