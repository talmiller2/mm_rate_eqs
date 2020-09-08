import numpy as np


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


def theta_fun(x, x0=0):
    """
    theta or relu function
    """
    if x > x0:
        return x
    else:
        return x0


def theta_fun_generic(x, x0=0):
    """
    theta function that can operate both on numbers os arrays
    """
    if is_array_like(x):
        y = np.zeros(len(x))
        for i in range(len(x)):
            y[i] = theta_fun(x[i], x0=x0)
        return y
    else:
        return theta_fun(x, x0=x0)
