import numpy as np


def stack_all_flattened(*args: np.ndarray) -> np.ndarray:
    """
    Horizontally stack all arrays provided.

    Arrays are flattened before stacking.
    """
    return np.hstack(tuple(arg.flatten() for arg in args))
