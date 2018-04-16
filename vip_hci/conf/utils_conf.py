#! /usr/bin/env python

"""
Module with utilities.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = []

import sys
import numpy as np

sep = '-' * 80


def check_array(input, dim=1, name=None):
    """ Checks the dimensionality of input. Returns it as a np.ndarray.
    """
    if name is None:
        name = 'Input'

    if dim == 1:
        msg = name + ' must be either a list or a 1d np.ndarray'
        if isinstance(input, (list, tuple)):
            input = np.array(input)
        if not isinstance(input, np.ndarray):
            raise TypeError(msg)
        if not input.ndim == 1:
            raise TypeError(msg)
    elif dim == 2:
        msg = name + ' must be an image or 2d np.ndarray'
        if not isinstance(input, np.ndarray):
            if not input.ndim == 2:
                raise TypeError(msg)
    elif dim == 3:
        msg = name + ' must be a cube or 3d np.ndarray'
        if not isinstance(input, np.ndarray):
            if not input.ndim == 3:
                raise TypeError(msg)
    elif dim == 4:
        msg = name + ' must be a cube or 4d np.ndarray'
        if not isinstance(input, np.ndarray):
            if not input.ndim == 4:
                raise TypeError(msg)
    return np.array(input)


def eval_func_tuple(f_args):
    """ Takes a tuple of a function and args, evaluates and returns result"""
    return f_args[0](*f_args[1:])                       


class redirect_output(object):
    """ Context manager for redirecting stdout/err to files"""
    def __init__(self, stdout='', stderr=''):
        self.stdout = stdout
        self.stderr = stderr

    def __enter__(self):
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr

        if self.stdout:
            sys.stdout = open(self.stdout, 'w')
        if self.stderr:
            if self.stderr == self.stdout:
                sys.stderr = sys.stdout
            else:
                sys.stderr = open(self.stderr, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.sys_stdout
        sys.stderr = self.sys_stderr

