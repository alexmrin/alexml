import inspect

import numpy as np

def check_type(variable, expected_type, allow_none=False):
    """
    Checks if a variable is of an expected type.

    Args:
        variable: The variable to check.
        expected_type: The expected type for the variable.
        allow_none (bool): Flag to allow None as a valid value for the variable.

    Raises:
        TypeError: If the variable is not of the expected type and not None (if allow_none is True).
    """
    if not isinstance(variable, expected_type) and not allow_none and variable is not None and not (inspect.isclass(variable) and issubclass(variable, expected_type)):
        raise TypeError(f"Expected type {expected_type.__name__}, got type {type(variable).__name__} instead.")
    
def exponential_moving_average(arr: np.ndarray, alpha: float):
    """
    Computes the Exponential Moving Average (EMA) of a given array.

    Args:
        arr (np.ndarray): A one-dimensional NumPy array of numerical values for 
                          which the EMA is to be calculated.
        alpha (float): The smoothing factor applied to the EMA, within the range (0, 1). 
                       A higher alpha discounts older observations faster. 
    Returns:
        np.ndarray: A NumPy array containing the EMA of the input array. Each element 
                    in the returned array corresponds to the EMA up to that point in 
                    the input array.
    """
    ema = [arr[0]]
    for i in range(1, arr.shape[0]):
        ema.append(arr[i] * alpha + ema[i-1] * (1 - alpha))
    return np.array(ema)