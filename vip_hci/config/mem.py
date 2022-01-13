#! /usr/bin/env python
"""
System memory related functions
"""

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['check_enough_memory',
           'get_available_memory']

from psutil import virtual_memory


def get_available_memory(verbose=True):
    """
    Get the available memory in bytes.

    Parameters
    ----------
    verbose : bool, optional
        Print out the total/available memory

    Returns
    -------
    available_memory : int
        The available memory in bytes.

    """
    mem = virtual_memory()
    if verbose:
        print("System total memory = {:.3f} GB".format(mem.total/1e9))
        print("System available memory = {:.3f} GB".format(mem.available/1e9))
    return mem.available


def check_enough_memory(input_bytes, factor=1, raise_error=True, error_msg='',
                        verbose=True):
    """
    Check if ``input_bytes`` are larger than system's available memory times
    ``factor``. This function is used to check the inputs (largest ones such as
    multi-dimensional cubes) of algorithms and avoid system/Python crashes or
    heavy swapping.

    Parameters
    ----------
    input_bytes : float
        The size in bytes of the inputs of a given function.
    factor : float, optional
        Scales how much memory is needed in terms of the size of input_bytes.
    raise_error : bool, optional
        If True, a RuntimeError is raised when the condition is not met.
    error_msg : str, optional
        [raise_error=True] To be appended to the message of the RuntimeError.
    verbose : bool, optional
        If True, information about the available memory is printed out.

    """
    available_memory = get_available_memory(verbose=verbose)
    if input_bytes > factor * available_memory:
        if raise_error:
            raise RuntimeError('Input is larger than available system memory' +
                               error_msg)
        return False
    else:
        return True
