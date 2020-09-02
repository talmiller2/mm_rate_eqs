import numpy as np


def theta_fun(x):
    if x > 0:
        return x
    else:
        return 0


def is_array_like(obj):
    '''
    Check if an object is an array with more than a single entry
    '''
    is_normal_array_like = isinstance(obj, (list, tuple))
    is_numpy_array_like = isinstance(obj, (np.ndarray)) and obj.shape is not ()
    if is_normal_array_like or is_numpy_array_like:
        return True
    else:
        return False
