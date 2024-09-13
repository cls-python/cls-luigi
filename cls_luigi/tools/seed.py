from typing import Optional, Tuple
import random


def set_seed(
    seed: Optional[int],
    radom_seed_interval: Tuple[int, int]
) -> Optional[int]:
    """
    Sets seed for reproducibility. If no seed in provided, it selects a random seed between the interval provided
    and returns it.

    Parameters
    ----------
    seed: Optional[int]
        The seed to be set. If None, it will select a random seed.
    radom_seed_interval: Tuple[int, int]
        The interval from which the seed will be selected if seed is None.

    Returns
    -------
    Optional[int]
        The seed that was set, incase it was randomly selected.
    """
    return_seed = False

    if seed is None:
        return_seed = True
        seed = random.randint(*sorted(radom_seed_interval, reverse=False))

    random.seed(seed)

    try:
        import numpy
        numpy.random.seed(seed)
    except ImportError:
        print("Numpy not found. Skipping seed setting for numpy.")

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    except ImportError:
        print("Torch not found. Skipping seed setting for torch.")

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        print("Tensorflow not found. Skipping seed setting for tensorflow.")

    if return_seed is True:
        return seed
