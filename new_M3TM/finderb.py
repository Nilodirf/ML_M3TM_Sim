import numpy as np


def finderb(key, array):
    """finderb

    Binary search algorithm for sorted array. Searches for the first index
    ``i`` of array where ``key`` >= ``array[i]``. ``key`` can be a scalar or
    a np.ndarray of keys. ``array`` must be a sorted np.ndarray.

    Author: André Bojahr.
    Licence: BSD.

    Args:
        key (float, ndarray[float]): single or multiple sorted keys.
        array (ndarray[float]): sorted array.

    Returns:
        i (ndarray[float]): position indices for each key in the array.

    """
    key = np.array(key, ndmin=1)
    n = len(key)
    i = np.zeros([n], dtype=int)

    for m in range(n):
        i[m] = finderb_nest(key[m], array)
    return i


def finderb_nest(key, array):
    """finderb_nest

    Nested sub-function of :func:`.finderb` for one single key.

    Author: André Bojahr.
    Licence: BSD.

    Args:
        key (float): single key.
        array (ndarray[float]): sorted array.

    Returns:
        a (float): position index of key in the array.

    """
    a = 0  # start of intervall
    b = len(array)  # end of intervall

    # if the key is smaller than the first element of the
    # vector we return 1
    if key < array[0]:
        return 0

    while (b-a) > 1:  # loop until the intervall is larger than 1
        c = int(np.floor((a+b)/2))  # center of intervall
        if key < array[c]:
            # the key is in the left half-intervall
            b = c
        else:
            # the key is in the right half-intervall
            a = c

    return a

