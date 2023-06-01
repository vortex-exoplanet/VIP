"""Module for various object oriented functions."""
from collections import OrderedDict
from dataclasses import dataclass
from inspect import signature
from typing import Callable

import numpy as np


def filter_duplicate_keys(
    filter_item: object | dict, ref_item: object | dict, filter_in: bool = True
):
    """
    Filter in or out keys of an item based on a reference item.

    Can keep only the keys from a reference item or exclude all of them, based
    on the boolean `filter_in`.

    Parameters
    ----------
    filter_item : Object or dict
        An object or a dictionnary that needs to be filtered.
    ref_item : Object or dict
        An object or a dictionnary that gives the keys wanted for filtering.
    filter_in : bool
        If True, keeps only the keys from reference, else erase only those keys.
    """

    # Conversion to dictionnary if the items are objects
    if isinstance(filter_item, dict):
        filter_dict = filter_item
    elif isinstance(filter_item, object):
        filter_dict = vars(filter_item)
    else:
        raise TypeError("The item to be filtered is neither a dictionnary or an object")

    if isinstance(ref_item, dict):
        ref_dict = ref_item
    elif isinstance(ref_item, object):
        ref_dict = vars(ref_item)
    else:
        raise TypeError("The reference item is neither a dictionnary or an object")

    # Find keys that must serve as a filter for `filter_item`
    common_keys = set(filter_dict.keys()) & set(ref_dict.keys())
    # Remove the duplicate keys in the `obj_params`

    if filter_in:
        for key in common_keys:
            del filter_dict[key]
    else:
        filter_dict = {
            key: value for key, value in filter_dict.items() if key in ref_dict.keys()
        }

    return filter_dict


def setup_parameters(
    params_obj: object,
    fkt: Callable,
    as_list: bool = False,
    **add_params: dict,
) -> dict | list:
    """
    Help creating a dictionnary of parameters for a given function.

    Look for the exact list of parameters needed for the ``fkt`` function and takes
    only the attributes needed from the ``params_obj``. More parameters can be
    included with the ``**add_pararms`` dictionnary.

    Parameters
    ----------
    params_obj : object
        Parameters to sort and order for the function.
    fkt : function
        The function we want to give parameters to.
    as_list : boolean
        Determines if the set of parameters should be return as a list instead
        of a dictionnary.
    **add_params : dictionnary, optional
        Additionnal parameters that may not be included in the params_obj.

    Returns
    -------
    params_setup : dictionnary or list
        The dictionnary comprised of parameters needed for the function, selected
        amongst attributes of PostProc objects and additionnal parameters. Can
        be a list if asked for (used in specific cases such as when calling
        functions through ``vip_hci.config.utils_conf.pool_map``, see an example
        in ``vip_hci.psfsub.framediff``).

    """
    wanted_params = OrderedDict(signature(fkt).parameters)
    # Remove dupe keys in params_obj from add_params
    if add_params is not None:
        obj_params = filter_duplicate_keys(filter_item=params_obj, ref_item=add_params)
        all_params = {**obj_params, **add_params}
    else:
        all_params = obj_params

    params_setup = OrderedDict(
        (param, all_params[param]) for param in wanted_params if param in all_params
    )

    if params_obj.verbose:
        print(f"The following parameters will be used for the run of {fkt.__name__} :")
        print_algo_params(params_setup)

    # For *args support, if an ordered list of parameters is needed
    if as_list:
        params_setup = list(params_setup.values())

    return params_setup


def print_algo_params(function_parameters) -> None:
    """Print the parameters that will be used for the run of an algorithm."""
    for key, value in function_parameters.items():
        if isinstance(value, np.ndarray) or isinstance(value, list):
            print(f"- {key} : np.ndarray or list (not shown)")
        else:
            print(f"- {key} : {value}")
