#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module with utilities.
"""

__author__ = 'Carlos Alberto Gomez Gonzalez, Ralf Farkas'
__all__ = ['Progressbar',
           'check_array',
           'sep',
           'vip_figsize',
           'vip_figdpi']

import os
import sys
import numpy as np

import itertools as itt
from inspect import getargspec
from functools import wraps
import multiprocessing
from vip_hci import __version__

sep = '―' * 80
vip_figsize = (8, 5)
vip_figdpi = 100


def print_precision(array, precision=3):
    """ Prints an array with a given floating point precision. 3 by default.
    """
    return print(np.array2string(array.astype(float), precision=precision))


class SaveableEmpty(object):
    """
    Empty object. Used by ``Saveable`` to restore the state of an object without
    calling __init__. Similar to what pickle/copy do.
    """
    pass


class Saveable(object):
    def save(self, filename):
        """
        Save a VIP object to a npz file.


        """

        vip_object = self.__class__.__name__

        if hasattr(self, "_saved_attributes"):

            data = {}

            for a in self._saved_attributes:
                if hasattr(self, a):
                    data[a] = getattr(self, a)

                    # set marker to re-build the original datatype
                    # (for non-np types like float, string, ...)
                    if not isinstance(getattr(self, a), np.ndarray):
                        data["_item_{}".format(a)] = True

                np.savez_compressed(filename, _vip_version=__version__,
                                    _vip_object=vip_object, **data)

        else:
            raise RuntimeError("_saved_attributes not found for class {}"
                               "".format(vip_object))

    @classmethod
    def load(cls, filename):
        try:
            data = np.load(filename, allow_pickle=True)
        except:
            data = np.load(filename + ".npz", allow_pickle=True)

        if "_vip_object" not in data:
            raise RuntimeError("The file you specified is not a VIP object.")

        file_vip_object = data["_vip_object"].item()
        if file_vip_object != cls.__name__:
            raise RuntimeError("The object in the file is of type '{}', please "
                               "use that classes 'load()' method instead."
                               "".format(file_vip_object))

        file_vip_version = data["_vip_version"].item()
        if file_vip_version != __version__:
            print("The file was saved with VIP {}. There may be some"
                  "compatibility issues. Use with care."
                  "".format(file_vip_version))

        self = SaveableEmpty()
        self.__class__ = cls

        for k in data:
            if k.startswith("_"):
                continue

            if "_item_{}".format(k) in data:
                setattr(self, k, data[k].item())  # un-pack np array
            else:
                setattr(self, k, data[k])

        # add non-saved, but expected attributes (backwards compatibility)
        for exp_k in self._saved_attributes:
            if exp_k not in data:
                setattr(self, exp_k, None)

        return self


class Progressbar(object):
    """ Show progress bars. Supports multiple backends.

    Examples
    --------
    .. code:: python

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


class NoProgressbar(object):
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


def algo_calculates_decorator(*calculated_attributes):
    """
    Decorator for HCIPostProcAlgo methods, describe what they calculate.
    
    There are three benefits from decorating a method:
    
    - if ``verbose=True``, prints a message about the calculated attributes and
      the ones which can be calculated next.
    - the attributes which *can* be calculated by this method are tracked, so
      if a user tries to access them *before* the function is called, an
      informative error message can be shown
    - the object knows which attributes to reset when ``run()`` is called a
      second time, on a different dataset.

    Parameters
    ----------
    *calculated_attributes : list of strings
        Strings denominating the attributes the decorated function calculates.
    
    Examples
    --------
    
    .. code:: python

        from .conf import algo_calculates_decorator as calculates

        class HCIMyAlgo(HCIPostPRocAlgo):
            def __init__(self, my_algo_param):
                self.store_args(locals())
            
            @calculates("final_frame", "snr_map")
            def run(dataset=None, verbose=True):
                frame, snr = my_heavy_calculation()
                
                self.final_frame = frame
                self.snr_map = snr

    """
    def decorator(fkt):
        @wraps(fkt)
        def wrapper(self, *args, **kwargs):
            # run the actual method
            res = fkt(self, *args, **kwargs)

            # get the kwargs the fkt sees. Note that this is a combination of
            # the *default* kwargs and the kwargs *passed* by the user
            a = getargspec(fkt)
            all_kwargs = dict(zip(a.args[-len(a.defaults):], a.defaults))
            all_kwargs.update(kwargs)
            
            if not hasattr(self, "_called_calculators"):
                self._called_calculators = []
            self._called_calculators.append(fkt.__name__)

            # show help message
            if all_kwargs.get("verbose", False):
                self._show_attribute_help(fkt.__name__)

            return res

        # set an attribute on the wrapper so _get_calculations() can find it:
        wrapper._calculates = calculated_attributes
        return wrapper
    return decorator


def check_array(input_array, dim, msg=None):
    """ Checks the dimensionality of input. In case the check is not successful,
    a TypeError is raised.

    Parameters
    ----------
    input_array : list, tuple or np.ndarray
        Input data.
    dim : int or tuple
        Number of dimensions that ``input_array`` should have. ``dim`` can take
        one of these values: 1, 2, 3, 4, (1,2), (2,3), (3,4) or (2,3,4).
    msg : str, optional
        String to be used in the error message (``input_array`` name).

    """
    if not isinstance(input_array, (list, tuple, np.ndarray)):
        raise TypeError("`input_array` must be a list, tuple of numpy ndarray")

    if msg is None:
        msg = 'Input array'
    else:
        msg = '`' + msg + '`'

    error_msg = "`dim` must be: 1, 2, 3, 4, (1,2), (2,3), (3,4) or (2,3,4)"
    if isinstance(dim, int):
        if dim < 1 or dim > 4:
            raise ValueError(error_msg)
    elif isinstance(dim, tuple):
        if dim not in ((1,2), (2,3), (3,4), (2,3,4)):
            raise ValueError(error_msg)

    msg2 = ' must be a '
    msg3 = 'd numpy ndarray'

    if dim == 1:
        if isinstance(input_array, (list, tuple)):
            input_array = np.array(input_array)
        if not isinstance(input_array, np.ndarray):
            raise TypeError(msg + msg2 + 'list, tuple or a 1' + msg3)
        if not input_array.ndim == dim:
            raise TypeError(msg + msg2 + 'list, tuple or a 1' + msg3)

    elif dim in (2, 3, 4):
        if not isinstance(input_array, np.ndarray):
            raise TypeError(msg + msg2 + str(dim) + msg3)
        else:
            if not input_array.ndim == dim:
                raise TypeError(msg + msg2 + str(dim) + msg3)

    elif isinstance(dim, tuple):
        if dim == (1, 2):
            msg_tup = '1 or 2'
        elif dim == (2, 3):
            msg_tup = '2 or 3'
        elif dim == (3, 4):
            msg_tup = '3 or 4'
        elif dim == (2, 3, 4):
            msg_tup = '2, 3 or 4'

        if isinstance(input_array, np.ndarray):
            if input_array.ndim not in dim:
                raise TypeError(msg + msg2 + msg_tup + msg3)
        else:
            raise TypeError(msg + msg2 + msg_tup + msg3)


def frame_or_shape(data):
    """
    Sanitize ``data``, always return a 2d frame.

    If ``data`` is a 2d frame, it is returned unchanged. If it is a shaped,
    return an empty array of that shape.

    Parameters
    ----------
    data : 2d ndarray or shape tuple

    Returns
    -------
    array : 2d ndarray

    """
    if isinstance(data, np.ndarray):
        array = data
        if array.ndim != 2:
            raise TypeError('`data` is not a frame or 2d array')
    elif isinstance(data, tuple):
        array = np.zeros(data)
    else:
        raise TypeError('`data` must be a tuple (shape) or a 2d array')

    return array


def eval_func_tuple(f_args):
    """ Takes a tuple of a function and args, evaluates and returns result"""
    return f_args[0](*f_args[1:])


class FixedObj(object):
    def __init__(self, v):
        self.v = v


def iterable(v):
    """ Helper function for ``pool_map``: prevents the argument from being
    wrapped in ``itertools.repeat()``.

    Examples
    --------
    .. code-block:: python

        # we have a worker function whic processes a word:

        def worker(word, method):
            # ...

        # we want to process these words in parallel fasion:
        words = ["lorem", "ipsum", "esse", "ea", "eiusmod"]

        # but all with
        method = 1

        # we then would use
        pool_map(3, worker, iterable(words), method)

        # this results in calling
        #
        # worker(words[0], 1)
        # worker(words[1], 1)
        # worker(words[2], 1)
        # ...
    """
    return FixedObj(v)


def pool_map(nproc, fkt, *args, **kwargs):
    """
    Abstraction layer for multiprocessing. When ``nproc=1``, the builtin
    ``map()`` is used. For ``nproc>1`` a ``multiprocessing.Pool`` is created.

    Parameters
    ----------
    nproc : int
        Number of processes to use.
    fkt : callable
        The function to be called with each ``*args``
    *args : function arguments
        Arguments passed to ``fkt`` By default, ``itertools.repeat`` is applied
        on all the arguments, except when you wrap the argument in
        ``iterable()``.
    msg : str or None, optional
        Description to be displayed.
    progressbar_single : bool, optional
        Display a progress bar when single-processing is used. Defaults to
        ``False``.
    verbose : bool, optional
        Show more output. Also disables the progress bar when set to ``False``.

    Returns
    -------
    res : list
        A list with the results.

    """
    multiprocessing.set_start_method('fork', force=True)
    from multiprocessing import Pool

    msg = kwargs.get("msg", None)
    verbose = kwargs.get("verbose", True)
    progressbar_single = kwargs.get("progressbar_single", False)
    _generator = kwargs.get("_generator", False)  # not exposed in docstring

    args_r = [a.v if isinstance(a, FixedObj) else itt.repeat(a) for a in args]
    z = zip(itt.repeat(fkt), *args_r)

    if nproc == 1:
        if progressbar_single:
            total = len([a.v for a in args if isinstance(a, FixedObj)][0])
            z = Progressbar(z, desc=msg, verbose=verbose, total=total)
        res = map(eval_func_tuple, z)
        if not _generator:
            res = list(res)
    else:
        if verbose and msg is not None:
            print("{} with {} processes".format(msg, nproc))
        pool = Pool(processes=nproc)
        if _generator:
            res = pool.imap(eval_func_tuple, z)
        else:
            res = pool.map(eval_func_tuple, z)
        pool.close()
        pool.join()

    return res


def pool_imap(nproc, fkt, *args, **kwargs):
    """
    Generator version of ``pool_map``. Useful when showing a progress bar for
    multiprocessing (see examples).

    Parameters
    ----------
    nproc : int
        Number of processes to use.
    fkt : callable
        The function to be called with each ``*args``
    *args : function arguments
        Arguments passed to ``fkt``
    msg : str or None, optional
        Description to be displayed.
    progressbar_single : bool, optional
        Display a progress bar when single-processing is used. Defaults to
        ``True``.
    verbose : bool, optional
        Show more output. Also disables the progress bar when set to ``False``.

    Examples
    --------

    .. code-block:: python

        # using pool_map

        res = pool_map(2, my_worker_function, *args)

        # using pool_imap with a progessbar:

        res = list(Progressbar(pool_imap(2, my_worker_function, *args)))

    """
    kwargs["_generator"] = True
    return pool_map(nproc, fkt, *args, **kwargs)


def repeat(*args):
    """
    Applies ``itertools.repeat`` to every ``args``.


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
    Chunks a list into ``n`` parts. The order of ``l`` is not kept. Useful for
    parallel processing when a single call is too fast, so the overhead from
    managing the processes is heavier than the calculation itself.

    Parameters
    ----------
    l : list
        Input list.
    n : int
        Number of parts.

    Examples
    --------
    .. code-block:: python

        make_chunks(range(13), 3)
            # -> [[0, 3, 6, 9, 12], [1, 4, 7, 10], [2, 5, 8, 11]]
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
        # TODO: close self.stdout and self.stderr
        sys.stdout = self.sys_stdout
        sys.stderr = self.sys_stderr


def lines_of_code():
    """ Calculates the lines of code for VIP. Not objective measure of
    developer's work! (note to self).
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
