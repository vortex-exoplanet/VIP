#! /usr/bin/env python

"""
Module for functions that check available memory, and input sizes.
"""
from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['check_enough_memory',
           'get_available_memory']

from psutil import virtual_memory


def get_available_memory(verbose=True):
    """ Returns the available memory in bytes.
    """
    mem = virtual_memory()
    if verbose: 
        print("System total memory = {:.3f} GB".format(mem.total/1e9))
        print("System available memory = {:.3f} GB".format(mem.available/1e9))
    return mem.available


def check_enough_memory(input_bytes, factor=1, verbose=True):
    """ Checks if the system's available memory is larger than 
    factor*input_bytes. This function is used to check the inputs of algorithms
    and avoid system/Python crashes or heavy swapping.
    
    Parameters
    ----------
    input_bytes : float
        The size in bytes of the inputs of a given function.
    factor : float
        Scales how much memory is needed in terms of the size of input_bytes.
    """
    mem = virtual_memory()
    load = factor*input_bytes
    if load > mem.total:
        return False        # total physical memory is smaller than input
    if verbose:  
        print("System available memory = {} bytes".format(mem.available))
    if load >= mem.available:
        return False        # available memory is not enough
    else:
        return True         # enough available memory