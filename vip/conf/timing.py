#! /usr/bin/env python

"""
Module for timing functions.
"""

__author__ = 'C. Gomez @ ULg'
__all__ = ['timeInit', 'timing', 'timeFini']

from datetime import datetime

def timeInit():
    """Sets and prints the time in which the script started.
    
    Returns
    -------
    start_time : string
        Starting time.
    """
    print "-------------------------------------------------------------------"
    start_time = datetime.now()
    print "Starting time: " + start_time.strftime("%Y-%m-%d %H:%M:%S")
    print "-------------------------------------------------------------------"
    return start_time

def timing(start_time):
    """Prints the execution time of a script. It requires the initialization 
    with the function timeInit().
    """
    print "Running time:  " + str(datetime.now()-start_time)
    print "-------------------------------------------------------------------"

def timeFini(start_time):
    """Returns the execution time of a script. It requires the initialization 
    with the function timeInit().
    """
    return str(datetime.now()-start_time)

   
