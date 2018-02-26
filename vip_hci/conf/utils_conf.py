#! /usr/bin/env python

"""
Module with utilities.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = []

import sys


sep = '----------------------------------------------------------------------'

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
        
    
