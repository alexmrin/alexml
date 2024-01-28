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
    if not isinstance(variable, expected_type) and not (allow_none and variable is None):
        raise TypeError(f"Expected type {expected_type.__name__}, got {type(variable).__name__} instead.")