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
