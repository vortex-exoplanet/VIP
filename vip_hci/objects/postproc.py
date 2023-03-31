#! /usr/bin/env python
"""
Module containing basic classes for manipulating post-processing algorithms.

This includes the core PostProc class, parent to every algorithm object implementation,
but also the PPResult class, a container for the results obtained through those said
algorithm objects. PPResult is provided with the Session dataclass, which defines the
type of data stored from the results.
"""

__author__ = "Thomas Bédrine, Carlos Alberto Gomez Gonzalez, Ralf Farkas"
__all__ = ["PostProc", "PPResult", "ALL_SESSIONS", "LAST_SESSION"]

import pickle
import inspect
from dataclasses import dataclass, field
from typing import (
    Tuple,
    Union,
    Optional,
    NoReturn,
    Callable,
    List,
)

import numpy as np
from hciplot import plot_frames
from sklearn.base import BaseEstimator

from .dataset import Dataset
from ..metrics import snrmap
from ..config.utils_conf import algo_calculates_decorator as calculates
from ..config.utils_conf import Saveable

PROBLEMATIC_ATTRIBUTE_NAMES = ["_repr_html_"]
LAST_SESSION = -1
ALL_SESSIONS = -2
DATASET_PARAM = "dataset"


@dataclass
class Session:
    """
    Dataclass for post-processing information storage.

    Each session of post-processing with one of the PostProc objects has a defined set
    of parameters, a frame obtained with those parameters and a S/N map generated with
    that frame. The Session class holds them in case you need to access them later or
    compare with another session.
    """

    parameters: dict
    frame: np.ndarray
    snr_map: np.ndarray
    algo_name: str


# TODO: find a proper format for results saving (pdf, images, dictionnaries...)
# TODO: add a way to identify which algorithm was used
@dataclass
class PPResult(Saveable):
    """
    Container for results of post-processing algorithms.

    For each given set of data and parameters, a frame is computed by the PostProc
    algorithms, as well as a S/N map associated. To keep track of each of them, this
    object remembers each set of parameters, frame and S/N map as a session. Sessions
    are numbered in order of creation from 0 to X, and they are displayed to the user
    as going from 1 to X+1.
    """

    sessions: List = field(default_factory=lambda: [])

    def register_session(
        self,
        frame: np.ndarray,
        algo_name: Optional[dict] = None,
        params: Optional[dict] = None,
        snr_map: Optional[np.ndarray] = None,
    ) -> None:
        """
        Register data for a new session or updating data for an existing one.

        Parameters
        ----------
        frame : np.ndarray
            Frame obtained after an iteration of a PostProc object.
        params : dictionnary, optional
            Set of parameters used for an iteration of a PostProc object.
        snr_map : np.ndarray, optional
            Signal-to-noise ratio map generated through the ``make_snrmap`` method of
            PostProc. Usually given after generating ``frame``.

        """
        # If frame is already registered in a session, add the associated snr_map only
        for session in self.sessions:
            if (frame == session.frame).all():
                session.snr_map = snr_map
                return

        # TODO: review filter_params to only target cube and angles, not all ndarrays
        # TODO: rename angles-type parameters in all procedural functions
        # Otherwise, register a new session
        filter_params = {
            key: params[key]
            for key in params
            if not isinstance(params[key], np.ndarray)
        }
        new_session = Session(
            parameters=filter_params,
            frame=frame,
            snr_map=snr_map,
            algo_name=algo_name,
        )
        self.sessions.append(new_session)

    def show_session_results(
        self,
        session_id: Optional[int] = LAST_SESSION,
        label: Optional[Union[Tuple[str], bool]] = True,
    ) -> None:
        """
        Print the parameters and plot the frame (and S/N map if able) of a session(s).

        Parameters
        ----------
        session_id : int, list of int or str, optional
            The ID of the session(s) to show. It is possible to get several sessions
            results by giving a list of int or "all" to get all of them. By default,
            the last session is displayed (index -1).

        """
        if self.sessions:
            if isinstance(session_id, list):
                if all(isinstance(s_id, int) for s_id in session_id):
                    for s_id in session_id:
                        self._show_single_session(s_id, label)
            elif session_id == ALL_SESSIONS:
                for s_id, _ in enumerate(self.sessions):
                    self._show_single_session(s_id, label)
            elif session_id in range(ALL_SESSIONS + 1, len(self.sessions)):
                self._show_single_session(session_id, label)
            else:
                raise ValueError(
                    "Given session ID isn't an integer. Please give an integer or a"
                    "list of integers (includes constant values such as ALL_SESSIONS or"
                    " LAST_SESSION)."
                )
        else:
            raise AttributeError(
                "No session was registered yet. Please register"
                " a session with the function `register_session`."
            )

    def _show_single_session(
        self,
        session_id: Optional[int],
        label: Optional[Union[Tuple[str], bool]] = True,
    ) -> None:
        """
        Display an individual session.

        Used a sub function to be called by ``show_session_results``.

        Parameters
        ----------
        session_id : int, optional
            Number of the session to be displayed.
        label : tuple of str or bool, optional
            Defines the label given to the frames plotted. If True, prints the default
            label for each frame, if False, prints nothing. Instead if the label is a
            tuple of str, sets them as the label for each frame.

        """
        if session_id == LAST_SESSION:
            session_label = "last session"
        else:
            session_label = "session n°" + str(session_id + 1)
        print(
            "Parameters used for the",
            session_label,
            f"(function used : {self.sessions[session_id].algo_name}) : ",
        )
        print_algo_params(self.sessions[session_id].parameters)

        if isinstance(label, bool):
            if label:
                _frame_label = "Frame obtained for the " + session_label
                _snr_label = "S/N map obtained for the " + session_label
            else:
                _frame_label = ""
                _snr_label = ""
        else:
            _frame_label, _snr_label = label

        if self.sessions[session_id].snr_map is not None:
            plot_frames(
                (
                    self.sessions[session_id].frame,
                    self.sessions[session_id].snr_map,
                ),
                label=(_frame_label, _snr_label),
            )
        else:
            plot_frames(self.sessions[session_id].frame, label=_frame_label)


@dataclass
class PostProc(BaseEstimator):
    """
    Base post-processing algorithm class.

    Does not need an ``__init__`` because as a parent class for every algorithm object,
    there is no reason to create a PostProc object. Inherited classes benefit from the
    ``dataclass_builder`` support for their initialization and no further methods are
    needed to create those.

    The PostProc is still very useful as it provides crucial utility common to all the
    inherited objects, such as :
        - establishing a list of attributes which need to be calculated
        - updating the dataset used for the algorithm if needed
        - calculating the signal-to-noise ratio map after a corrected frame has been
        generated
        - setting up parameters for the algorithm.

    """

    dataset: Dataset = None
    verbose: bool = True
    results: PPResult = None
    frame_final: np.ndarray = None

    def print_parameters(self) -> None:
        """Print out the parameters of the algorithm."""
        for key, value in self.__dict__.items():
            print(f"{key} : {value}")

    def _update_dataset(self, dataset: Optional[Dataset] = None) -> None:
        """
        Handle a dataset passed to ``run()``.

        It is possible to specify a dataset using the constructor, or using the
        ``run()`` function. This helper function checks that there is a dataset
        to work with.

        Parameters
        ----------
        dataset : Dataset or None, optional

        """
        if dataset is not None:
            print(
                "A new dataset was provided to run, all previous results were cleared."
            )
            self.dataset = dataset
            self._reset_results()
        elif self.dataset is None:
            raise AttributeError(
                "No dataset was specified ! Please give a valid dataset inside the"
                "builder of the associated algorithm or inside the `run()` function."
            )
        else:
            print("No changes were made to the dataset.")

    def get_params_from_results(self, session_id: int) -> None:
        """
        Copy a previously registered configuration from the results to the object.

        Parameters
        ----------
        session_id : int
            The ID of the session to load the configuration from.

        """
        if self.results is None:
            raise AttributeError(
                "No results were saved yet ! Please give the object a PPResult instance"
                " and run the object at least once."
            )

        res = self.results.sessions
        if session_id > len(res) or res == []:
            raise ValueError(
                f"ID is higher than the current number of sessions registered. "
                f"There are {len(self.results.sessions)} saved now.",
            )

        if res[session_id].algo_name not in self._algo_name:
            raise ValueError(
                "The function used for that session does not match your object."
                " Please choose a session with a corresponding function."
            )

        for key, value in res[session_id].parameters.items():
            setattr(self, key, value)
        print("Configuration loaded :")
        print_algo_params(res[session_id].parameters)

    # TODO : identify the problem around the element `_repr_html_`
    def _get_calculations(self) -> dict:
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
        for element in dir(self):
            # BLACKMAGIC : _repr_html_ must be skipped
            """
            `_repr_html_` is an element of the directory of the PostProc object which
            causes the search of calculated attributes to overflow, looping indefinitely
            and never reaching the actual elements containing those said attributes.
            It will be skipped until the issue has been properly identified and fixed.
            You can uncomment the block below to observe how the directory loops after
            reaching that element - acknowledging you are not skipping it.
            """
            if element not in PROBLEMATIC_ATTRIBUTE_NAMES:
                try:
                    # print(
                    #     "directory element : ",
                    #     e,
                    #     ", calculations list : ",
                    #     calculations,
                    # )
                    for k in getattr(getattr(self, element), "_calculates"):
                        calculations[k] = element
                except AttributeError:
                    pass

        return calculations

    def _reset_results(self) -> None:
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

    def __getattr__(self, attr: str) -> NoReturn:
        """
        ``__getattr__`` is only called when an attribute does *not* exist.

        Catching this event allows us to output proper error messages when an
        attribute was not calculated yet.
        """
        calculations = self._get_calculations()
        if attr in calculations:
            raise AttributeError(
                f"The {attr} was not calculated yet. Call {calculations[attr]} first."
            )
        # this raises a regular AttributeError:
        return self.__getattribute__(attr)

    def _show_attribute_help(self, function_name: Callable) -> None:
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
        for attr, func in calculations.items():
            if hasattr(self, attr) and function_name == func:
                print(f"\t{attr}")

        not_calculated_yet = [
            (a, f)
            for a, f in calculations.items()
            if (f not in self._called_calculators and not hasattr(self, a))
        ]
        if len(not_calculated_yet) > 0:
            print("The following attributes can be calculated now:")
            for attr, func in not_calculated_yet:
                print(f"\t{attr}\twith .{func}()")

    @calculates("snr_map", "detection_map")
    def make_snrmap(
        self,
        approximated: Optional[bool] = False,
        plot: Optional[bool] = False,
        known_sources: Optional[Union[Tuple, Tuple[Tuple]]] = None,
        nproc: Optional[int] = None,
    ) -> None:
        """
        Calculate a S/N map from ``self.frame_final``.

        Parameters
        ----------
        results : PPResult object, optional
            Container for the results of the algorithm. May hold the parameters used,
            as well as the ``frame_final`` (and the ``snr_map`` if generated).
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
            verbose=self.verbose,
        )

        self.detection_map = self.snr_map

        if self.results is not None:
            self.results.register_session(frame=self.frame_final, snr_map=self.snr_map)

    def save(self, filename: str) -> None:
        """
        Pickle the algo object and save it to disk.

        Note that this also saves the associated ``self.dataset``, in a
        non-optimal way.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @calculates("frame_final")
    def run(self) -> None:
        """
        Run the algorithm. Should at least set `` self.frame_final``.

        Note
        ----
        This is the required signature of the ``run`` call. Child classes can
        add their own keyword arguments if needed.
        """
        raise NotImplementedError

    # TODO: write test
    def _setup_parameters(self, fkt: Callable, **add_params: dict) -> dict:
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
        if self.verbose:
            print(
                f"The following parameters will be used for the run of {fkt.__name__} :"
            )
            print_algo_params(params_dict)
        return params_dict


def print_algo_params(params: dict) -> None:
    """
    Print the parameters that will be used for the run of an algorithm.

    Parameters
    ----------
    params : dict
        Dictionnary of the parameters that are sent to the function.

    """
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            print(f"- {key} : np.ndarray (not shown)")
        else:
            print(f"- {key} : {value}")
