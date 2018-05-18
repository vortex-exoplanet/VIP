#! /usr/bin/env python

"""
Module with utilities.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez, Ralf Farkas'
__all__ = ['Progressbar',
           'pool_map', 'pool_imap', 'fixed',
           'make_chunks']

import os
import sys
import numpy as np

import itertools as itt
from multiprocessing import Pool

sep = '-' * 80
vip_figsize = (10, 5)
vip_print_np = {'float_kind': '{0:.3f}'.format}


class Progressbar(object):
    """ Show progress bars. Supports multiple backends.

    Examples
    --------
    from vip_hci.var import Progressbar
    Progressbar.backend = "tqdm"

    from time import sleep

    for i in Progressbar(range(50)):
        sleep(0.02)

    # or:

    bar = Progressbar(total=50):
    for i in range(50):
        sleep(0.02)
        bar.update()

    # Progressbar can be disabled globally using
    Progressbar.backend = "hide"

    # or locally using the ``verbose`` keyword:
    Progressbar(iterable, verbose=False)

    Notes
    -----
    - `leave` keyword is natively supported by tqdm, support could be added to
      other backends too?

    """
    backend = "pyprind"

    def __new__(cls, iterable=None, desc=None, total=None, leave=True,
                backend=None, verbose=True):
        import sys

        if backend is None:
            backend = Progressbar.backend

        if not verbose:
            backend = "hide"

        if backend == "tqdm":
            from tqdm import tqdm
            return tqdm(iterable=iterable, desc=desc, total=total, leave=leave,
                        ascii=True, ncols=80, file=sys.stdout,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed"
                                   "}<{remaining}{postfix}]") # remove rate_fmt
        elif backend == "tqdm_notebook":
            from tqdm import tqdm_notebook
            return tqdm_notebook(iterable=iterable, desc=desc, total=total,
                                 leave=leave)
        elif backend == "pyprind":
            from pyprind import ProgBar, prog_bar
            ProgBar._adjust_width = lambda self: None  # keep constant width
            if iterable is None:
                return ProgBar(total, title=desc, stream=1)
            else:
                return prog_bar(iterable, title=desc, stream=1,
                                iterations=total)
        elif backend == "hide":
            return NoProgressbar(iterable=iterable)
        else:
            raise NotImplementedError("unknown backend")

    def set(b):
        Progressbar.backend = b


class NoProgressbar():
    """ Wraps an ``iterable`` to behave like ``Progressbar``, but without
    producing output.
    """
    def __init__(self, iterable=None):
        self.iterable = iterable

    def __iter__(self):
        return self.iterable.__iter__()

    def __next__(self):
        return self.iterable.__next__()

    def __getattr__(self, key):
        return self.iterable.key

    def update(self):
        pass


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
            raise TypeError(msg)
        else:
            if not input.ndim == 2:
                raise TypeError(msg)
    elif dim == 3:
        msg = name + ' must be a cube or 3d np.ndarray'
        if not isinstance(input, np.ndarray):
            raise TypeError(msg)
        else:
            if not input.ndim == 3:
                raise TypeError(msg)
    elif dim == 4:
        msg = name + ' must be a cube or 4d np.ndarray'
        if not isinstance(input, np.ndarray):
            raise TypeError(msg)
        else:
            if not input.ndim == 4:
                raise TypeError(msg)
    return np.array(input)


def eval_func_tuple(f_args):
    """ Takes a tuple of a function and args, evaluates and returns result"""
    return f_args[0](*f_args[1:])                       


class FixedObj():
    def __init__(self, v):
        self.v = v

def fixed(v):
    return FixedObj(v)


def pool_map(nproc, fkt, *args, msg=None, verbose=True,
                   progressbar_single=True, generator=False):
    """
    pool.map which uses `fixed()` for ...

    """
    
    args_r = [a.v if isinstance(a, FixedObj) else itt.repeat(a) for a in args]
    z = zip(itt.repeat(fkt), *args_r)

    if nproc == 1:
        if progressbar_single:
            total = len([a.v for a in args if isinstance(a, FixedObj)][0])
            z = Progressbar(z, desc=msg, verbose=verbose, total=total)

        res = map(eval_func_tuple, z)
        if not generator:
            res = list(res)


    else:
        if verbose and msg is not None:
            print(msg+" with {} processes".format(nproc))
        pool = Pool(processes=nproc)
        if generator:
            res = pool.imap(eval_func_tuple, z)
        else:
            res = pool.map(eval_func_tuple, z)
        pool.close()
        pool.join()

    return res


def pool_imap(nproc, fkt, *args, msg=None, verbose=True,
                   progressbar_single=True):
    return pool_map(nproc, fkt, *args, msg=msg, verbose=verbose,
                    progressbar_single=progressbar_single, generator=True)




def repeat(*args):
    """

    Examples
    --------

    # instead of using

    import itertools as itt
    
    my_fkt(itt.repeat(a), itt.repeat(b), itt.repeat(c), d, itt.repeat(e))

    # you could use `repeat`:

    my_fkt(*repeat(a, b, c), d, *repeat(e))

    """
    return [itt.repeat(a) for a in args]


def make_chunks(l, n):
    """
    make_chunks(range(13), 3)
        -> [[0, 3, 6, 9, 12], [1, 4, 7, 10], [2, 5, 8, 11]]
    """
    return [l[i::n] for i in range(n)]


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


def lines_of_code():
    """ Calculates the lines of code for VIP pipeline. Not objective measure
    of developer's work! (note to self).
    """
    cur_path = os.path.dirname(os.path.abspath(__file__))
    path = cur_path[:-len('conf')]

    ignore_set = set(["__init__.py"])

    loclist = []

    for pydir, _, pyfiles in os.walk(path):
        if 'exlib/' not in pydir:
            for pyfile in pyfiles:
                if pyfile not in ignore_set and pyfile.endswith(".py"):
                    totalpath = os.path.join(pydir, pyfile)
                    loclist.append(
                        (len(open(totalpath, "r").read().splitlines()),
                         totalpath.split(path)[1]))

    for linenumbercount, filename in loclist:
        print("{:05d} lines in {}".format(linenumbercount, filename))

    msg = "\nTotal: {} lines in ({}) excluding external libraries."
    print(msg.format(sum([x[0] for x in loclist]), path))
