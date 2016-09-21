#! /usr/bin/env python

"""
Module for timing functions.
"""

__author__ = 'C. Gomez @ ULg'
__all__ = ['timeInit', 'timing', 'timeFini']

from datetime import datetime
from .utils import sep

def timeInit(verbose=True):
    """Sets and prints the time in which the script started.
    
    Returns
    -------
    start_time : string
        Starting time.
    """
    start_time = datetime.now()
    if verbose:
        print sep
        print "Starting time: " + start_time.strftime("%Y-%m-%d %H:%M:%S")
        print sep
    return start_time

def timing(start_time):
    """Prints the execution time of a script. It requires the initialization 
    with the function timeInit().
    """
    print "Running time:  " + str(datetime.now()-start_time)
    print sep

def timeFini(start_time):
    """Returns the execution time of a script. It requires the initialization 
    with the function timeInit().
    """
    return str(datetime.now()-start_time)

   
