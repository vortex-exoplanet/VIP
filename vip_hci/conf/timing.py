#! /usr/bin/env python

"""
Functions for timing other functions/procedures.
"""
from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez'
__all__ = ['time_ini', 'timing', 'time_fin']

from datetime import datetime
from .utils_conf import sep

def time_ini(verbose=True):
    """Sets and prints the time in which the script started.
    
    Returns
    -------
    start_time : string
        Starting time.
    """
    start_time = datetime.now()
    if verbose:
        print(sep)
        print("Starting time: " + start_time.strftime("%Y-%m-%d %H:%M:%S"))
        print(sep)
    return start_time


def timing(start_time):
    """Prints the execution time of a script. It requires the initialization 
    with the function time_ini().
    """
    print("Running time:  " + str(datetime.now()-start_time))
    print(sep)


def time_fin(start_time):
    """Returns the execution time of a script. It requires the initialization 
    with the function time_ini().
    """
    return str(datetime.now()-start_time)

   
