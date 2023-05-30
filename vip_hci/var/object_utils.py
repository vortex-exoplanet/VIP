"""Module for various object oriented functions."""
from collections import OrderedDict
from dataclasses import dataclass
from inspect import signature
from typing import Callable

import numpy as np


@dataclass
class ParamsUtils:
    """Tools dedicated to handle the parameters of various functions."""

    function_parameters: dict = None

    def print_algo_params(self) -> None:
        """Print the parameters that will be used for the run of an algorithm."""
        for key, value in self.function_parameters.items():
            if isinstance(value, np.ndarray):
                print(f"- {key} : np.ndarray (not shown)")
            else:
                print(f"- {key} : {value}")

    def setup_parameters(
        self,
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
        params_dict : dictionnary or list
            The dictionnary comprised of parameters needed for the function, selected
            amongst attributes of PostProc objects and additionnal parameters. Can
            be a list if asked for (used in specific cases such as when calling
            functions through ``vip_hci.config.utils_conf.pool_map``, see an example
            in ``vip_hci.psfsub.framediff``).

        """
        wanted_params = OrderedDict(signature(fkt).parameters)
        obj_params = vars(params_obj)
        # Find keys that must be overridden by `add_params`
        common_keys = set(obj_params.keys()) & set(add_params.keys())
        # Remove the duplicate keys in the `obj_params`
        for key in common_keys:
            del obj_params[key]

        all_params = {**obj_params, **add_params}
        params_dict = OrderedDict(
            (param, all_params[param]) for param in wanted_params if param in all_params
        )

        self.function_parameters = params_dict
        if params_obj.verbose:
            print(
                f"The following parameters will be used for the run of {fkt.__name__} :"
            )
            self.print_algo_params()

        # For *args support, if an ordered list of parameters is needed
        if as_list:
            params_dict = list(params_dict.values())

        return params_dict
