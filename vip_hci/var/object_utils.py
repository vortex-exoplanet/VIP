"""Module for various object oriented functions."""
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
        self, params_obj: object, fkt: Callable, **add_params: dict
    ) -> dict:
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
        **add_params : dictionnary, optional
            Additionnal parameters that may not be included in the params_obj.

        Returns
        -------
        params_dict : dictionnary
            The dictionnary comprised of parameters needed for the function, selected
            amongst attributes of PostProc objects and additionnal parameters.

        """
        wanted_params = signature(fkt).parameters
        obj_params = vars(params_obj)
        # Find keys that must be overridden by `add_params`
        common_keys = set(obj_params.keys()) & set(add_params.keys())
        # Remove the duplicate keys in the `obj_params`
        for key in common_keys:
            del obj_params[key]

        all_params = {**obj_params, **add_params}
        params_dict = {
            param: all_params[param] for param in all_params if param in wanted_params
        }

        self.function_parameters = params_dict
        if params_obj.verbose:
            print(
                f"The following parameters will be used for the run of {fkt.__name__} :"
            )
            self.print_algo_params()

        return params_dict
