#! /usr/bin/env python
"""Module with the HCI<post-processing algorithms> classes."""

__author__ = "Thomas Bédrine, Carlos Alberto Gomez Gonzalez, Ralf Farkas"
__all__ = [
    "PostProc",
]

import pickle
import inspect
import numpy as np
from sklearn.base import BaseEstimator

from ..dataset import Dataset
from ..metrics import snrmap
from ..config.utils_conf import algo_calculates_decorator as calculates

PROBLEMATIC_ATTRIBUTE_NAMES = ["_repr_html_"]


class PostProc(BaseEstimator):
    """Base post-processing algorithm class."""

    def __init__(self, locals_dict, *skip):
        """
        Set up the algorithm parameters.

        This does multiple things:

        - verify that ``dataset`` is a Dataset object or ``None`` (it could
          also be provided to ``run``)
        - store all the keywords (from ``locals_dict``) as object attributes, so
          they can be accessed e.g. in the ``run()`` method
        - print out the full algorithm settings (user provided parameters +
          default ones) if ``verbose=True``

        Parameters
        ----------
        locals_dict : dict
            This should be ``locals()``. ``locals()`` contains *all* the
            variables defined in the local scope. Passed to
            ``self._store_args``.
        *skip : list of strings
            Passed on to ``self._store_args``. Refer to its documentation.

        Examples
        --------
        .. code:: python

            # when subclassing PostProc, make sure you call super()
            # with locals()! This means:

            class MySuperAlgo(PostProc):
                def __init__(self, algo_param_1=42, cool=True):
                    super(MySuperAlgo, self).__init__(locals())

                @calculates("frame")
                def run(self, dataset=None):
                    self.frame = 2 * self.algo_param_1

        """
        dataset = locals_dict.get("dataset", None)
        if not isinstance(dataset, (Dataset, type(None))):
            raise ValueError("`dataset` must be a Dataset object or None")

        self._store_args(locals_dict, *skip)

        verbose = locals_dict.get("verbose", True)
        if verbose:
            self._print_parameters()

    def _print_parameters(self):
        """Print out the parameters of the algorithm."""
        dicpar = self.get_params()
        for key in dicpar.keys():
            print("{}: {}".format(key, dicpar[key]))

    def _store_args(self, locals_dict, *skip):
        # TODO: this could be integrated with sklearn's BaseEstimator methods
        for k in locals_dict:
            if k == "self" or k in skip:
                continue
            setattr(self, k, locals_dict[k])

    def _get_dataset(self, dataset=None, verbose=True):
        """
        Handle a dataset passed to ``run()``.

        It is possible to specify a dataset using the constructor, or using the
        ``run()`` function. This helper function checks that there is a dataset
        to work with.

        Parameters
        ----------
        dataset : Dataset or None, optional
        verbose : bool, optional
            If ``True``, a message is printed out when a previous dataset was
            overwritten.

        Returns
        -------
        dataset : Dataset

        """
        if dataset is None:
            dataset = self.dataset
            if self.dataset is None:
                raise ValueError("no dataset specified!")
        else:
            if self.dataset is not None and verbose:
                print(
                    "a new dataset was provided to run(), all previous "
                    "results were cleared."
                )
            self.dataset = dataset
            self._reset_results()

        return dataset

    # TODO : identify the problem around the element `_repr_html_`
    def _get_calculations(self):
        """
        Get a list of all attributes which are *calculated*.

        This iterates over all the elements in an object and finds the functions
        which were decorated with ``@calculates`` (which are identified by the
        function attribute ``_calculates``). It then stores the calculated
        attributes, together with the corresponding method, and returns it.

        Returns
        -------
        calculations : dict
            Dictionary mapping a single "calculated attribute" to the method
            which calculates it.

        """
        calculations = {}
        for e in dir(self):
            # BLACKMAGIC : _repr_html_ must be skipped
            """
            `_repr_html_` is an element of the directory of the PostProc object which
            causes the search of calculated attributes to overflow, looping indefinitely
            and never reaching the actual elements containing those said attributes.
            It will be skipped until the issue has been properly identified and fixed.
            You can uncomment the block below to observe how the directory loops after
            reaching that element - acknowledging you are not skipping it.
            """
            if e not in PROBLEMATIC_ATTRIBUTE_NAMES:
                try:
                    # print(
                    #     "directory element : ",
                    #     e,
                    #     ", calculations list : ",
                    #     calculations,
                    # )
                    for k in getattr(getattr(self, e), "_calculates"):
                        calculations[k] = e
                except AttributeError:
                    pass

        return calculations

    def _reset_results(self):
        """
        Remove all calculated results from the object.

        By design, the PostProc's can be initialized without a dataset,
        so the dataset can be provided to the ``run`` method. This makes it
        possible to run the same algorithm on multiple datasets. In order not to
        keep results from an older ``run`` call when working on a new dataset,
        the stored results are reset using this function every time the ``run``
        method is called.
        """
        for attr in self._get_calculations():
            try:
                delattr(self, attr)
            except AttributeError:
                pass  # attribute/result was not calculated yet. Skip.

    def __getattr__(self, a):
        """
        ``__getattr__`` is only called when an attribute does *not* exist.

        Catching this event allows us to output proper error messages when an
        attribute was not calculated yet.
        """
        calculations = self._get_calculations()
        if a in calculations:
            raise AttributeError(
                "The '{}' was not calculated yet. Call '{}' "
                "first.".format(a, calculations[a])
            )
        else:
            # this raises a regular AttributeError:
            return self.__getattribute__(a)

    def _show_attribute_help(self, function_name):
        """
        Print information about the attributes a method calculated.

        This is called *automatically* when a method is decorated with
        ``@calculates``.

        Parameters
        ----------
        function_name : string
            The name of the method.

        """
        calculations = self._get_calculations()

        print("These attributes were just calculated:")
        for a, f in calculations.items():
            if hasattr(self, a) and function_name == f:
                print("\t{}".format(a))

        not_calculated_yet = [
            (a, f)
            for a, f in calculations.items()
            if (f not in self._called_calculators and not hasattr(self, a))
        ]
        if len(not_calculated_yet) > 0:
            print("The following attributes can be calculated now:")
            for a, f in not_calculated_yet:
                print("\t{}\twith .{}()".format(a, f))

    @calculates("snr_map", "detection_map")
    def make_snrmap(
        self,
        approximated=False,
        plot=False,
        known_sources=None,
        nproc=None,
        verbose=False,
    ):
        """
        Calculate a S/N map from ``self.frame_final``.

        Parameters
        ----------
        approximated : bool, optional
            If True, a proxy to the S/N calculation will be used. If False, the
            Mawet et al. 2014 definition is used.
        plot : bool, optional
            If True plots the S/N map. True by default.
        known_sources : None, tuple or tuple of tuples, optional
            To take into account existing sources. It should be a tuple of
            float/int or a tuple of tuples (of float/int) with the coordinate(s)
            of the known sources.
        nproc : int or None
            Number of processes for parallel computing.
        verbose: bool, optional
            Whether to print timing or not.

        Note
        ----
        This is needed for "classic" algorithms that produce a final residual
        image in their ``.run()`` method. To obtain a "detection map", which can
        be used for counting true/false positives, a SNR map has to be created.
        For other algorithms (like ANDROMEDA) which directly create a SNR or a
        probability map, this method should be overwritten and thus disabled.

        """
        if self.dataset.cube.ndim == 4:
            fwhm = np.mean(self.dataset.fwhm)
        else:
            fwhm = self.dataset.fwhm

        self.snr_map = snrmap(
            self.frame_final,
            fwhm,
            approximated,
            plot=plot,
            known_sources=known_sources,
            nproc=nproc,
            verbose=verbose,
        )

        self.detection_map = self.snr_map

    def save(self, filename):
        """
        Pickle the algo object and save it to disk.

        Note that this also saves the associated ``self.dataset``, in a
        non-optimal way.
        """
        pickle.dump(self, open(filename, "wb"))

    @calculates("frame_final")
    def run(self, dataset=None, nproc=1, verbose=True):
        """
        Run the algorithm. Should at least set `` self.frame_final``.

        Note
        ----
        This is the required signature of the ``run`` call. Child classes can
        add their own keyword arguments if needed.
        """
        raise NotImplementedError

    def _setup_parameters(self, fkt, **add_params):
        """
        Help creating a dictionnary of parameters for a given function.

        Look for the exact list of parameters needed for the ``fkt`` function and takes
        only the attributes needed from the PostProc project. More parameters can be
        included with the ``**add_pararms`` dictionnary.

        Parameters
        ----------
        fkt : function
            The function we want to give parameters to.
        **add_params : dictionnary, optional
            Additionnal parameters that may not be included in the PostProc object.

        Returns
        -------
        params_dict : dictionnary
            The dictionnary comprised of parameters needed for the function, selected
            amongst attributes of PostProc objects and additionnal parameters.

        """
        wanted_params = inspect.signature(fkt).parameters
        obj_params = vars(self)
        all_params = {**obj_params, **add_params}
        params_dict = {
            param: all_params[param] for param in all_params if param in wanted_params
        }
        return params_dict
