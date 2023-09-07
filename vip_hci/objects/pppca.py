#! /usr/bin/env python
"""Module for the post-processing PCA algorithm."""

__author__ = "Thomas Bédrine"
__all__ = ["PCABuilder", "PPPCA"]

from typing import Tuple, Optional, List
from dataclasses import dataclass, field

import numpy as np
from pandas import DataFrame
from dataclass_builder import dataclass_builder

from .dataset import Dataset
from .postproc import PostProc
from ..psfsub import (
    pca,
    pca_annular,
    pca_grid,
    pca_annulus,
    PCA_Params,
    PCA_ANNULAR_Params,
)
from ..config.paramenum import Adimsdi, ReturnList, Runmode
from ..config.utils_conf import algo_calculates_decorator as calculates
from ..config.utils_param import setup_parameters


@dataclass
class PPPCA(PostProc, PCA_Params, PCA_ANNULAR_Params):
    """
    Post-processing PCA algorithm, compatible with various options.

    Depending on what mode you need, parameters may vary. Check the list below to
    ensure which arguments are required. Currently, four variations of the PCA can be
    called :
        - full-frame PCA
        - annular PCA
        - grid PCA
        - single annulus PCA.
    Some parameters are common to several variations.

    Common parameters
    -----------------
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    _algo_name: str, optional
        Name of the algorithm wrapped by the object.

    Grid parameters
    ---------------
    range_pcs : None or tuple, optional
        The interval of PCs to be tried. Refer to ``vip_hci.psfsub.pca_grid`` for more
        information.
    mode : {'fullfr', 'annular'}, optional
        Mode for PCA processing (full-frame or just in an annulus).
    fmerit : {'px', 'max', 'mean'}
        The function of merit to be maximized. 'px' is *source_xy* pixel's SNR,
        'max' the maximum SNR in a FWHM circular aperture centered on
        ``source_xy`` and 'mean' is the mean SNR in the same circular aperture.
    plot : bool, optional
        Whether to plot the SNR and flux as functions of PCs and final PCA
        frame or not.
    save_plot: string
        If provided, the pc optimization plot will be saved to that path.
    initial_4dshape : None or tuple, optional
        Shape of the initial ADI+mSDI cube.
    exclude_negative_lobes : bool, opt
        Whether to include the adjacent aperture lobes to the tested location
        or not. Can be set to True if the image shows significant neg lobes.

    Single annulus parameters
    -------------------------
    r_guess : float
        Radius of the annulus in pixels.

    """

    # Common parameters/returns
    _algo_name: List[str] = field(
        default_factory=lambda: [
            "pca",
            "pca_annular",
            "pca_grid",
            "pca_annulus",
        ]
    )
    cube_sig: np.ndarray = None
    cube_residuals: np.ndarray = None
    cube_residuals_der: np.ndarray = None
    full_output = True
    # Full-frame returns
    pcs: np.ndarray = None
    cube_residuals_per_channel: np.ndarray = None
    cube_residuals_per_channel_der: np.ndarray = None
    cube_residuals_resc: np.ndarray = None
    final_residuals_cube: np.ndarray = None
    medians: np.ndarray = None
    # Grid parameters
    frames_final: np.ndarray = None
    range_pcs: Tuple[int] = None
    mode: str = "fullfr"
    fmerit: str = "mean"
    plot: bool = True
    save_plot: str = None
    exclude_negative_lobes: bool = False
    initial_4dshape: Tuple = None
    dataframe: DataFrame = None
    pc_list: List = None
    opt_number_pc: int = None
    # Single annulus parameters
    annulus_width: float = None  # Note: also used for Grid in annular mode
    r_guess: float = None

    # TODO: write test
    @calculates(
        "frame_final",
        "cube_reconstructed",
        "cube_residuals",
        "cube_residuals_der",
        "pcs",
        "cube_residuals_per_channel",
        "cube_residuals_per_channel_der",
        "cube_residuals_resc",
        "final_residuals_cube",
        "medians",
        "dataframe",
        "opt_number_pc",
    )
    def run(
        self,
        runmode: Optional[str] = Runmode.CLASSIC,
        dataset: Optional[Dataset] = None,
        nproc: Optional[int] = 1,
        verbose: Optional[bool] = True,
        full_output: Optional[bool] = True,
        **rot_options: Optional[dict],
    ):
        """
        Run the post-processing PCA algorithm for model PSF subtraction.

        Depending on the desired mode - full-frame or annular - parameters used will
        diverge, and calculated attributes may vary as well. In full-frame case :

            3D case:
                cube_reconstructed
                cube_residuals
                cube_residuals_der
            3D case, source_xy is None:
                cube_residuals
                pcs
            4D case, adimsdi="double":
                cube_residuals_per_channel
                cube_residuals_per_channel_der
            4D case, adimsdi="single":
                cube_residuals
                cube_residuals_resc

        Parameters
        ----------
        runmode : Enum, see ``vip_hci.config.paramenum.Runmode``
            Mode of execution for the PCA.
        dataset : Dataset, optional
            Dataset to process. If not provided, ``self.dataset`` is used (as
            set when initializing this object).
        nproc : int, optional
        verbose : bool, optional
            If True prints to stdout intermediate info.
        full_output: boolean, optional
            Whether to return the final median combined image only or with
            other intermediate arrays.
        rot_options: dictionary, optional
            Dictionary with optional keyword values for "border_mode", "mask_val",
            "edge_blend", "interp_zeros", "ker" (see documentation of
            ``vip_hci.preproc.frame_rotate``)

        """
        self.snr_map = None
        self._update_dataset(dataset)
        if self.dataset.fwhm is None:
            raise ValueError("`fwhm` has not been set")
        self._explicit_dataset()
        self.full_output = full_output
        match (runmode):
            case Runmode.CLASSIC:
                # TODO : review the wavelengths attribute to be a scale_list instead

                params_dict = self._create_parameters_dict(PCA_Params)

                all_params = {"algo_params": self, **rot_options}

                res = pca(**all_params)

                self._find_pca_mode(res=res)

                if self.results is not None and self.frame_final is not None:
                    self.results.register_session(
                        params=params_dict,
                        frame=self.frame_final,
                        algo_name=self._algo_name[0],
                    )

            case Runmode.ANNULAR:
                if self.nproc is None:
                    self.nproc = nproc

                params_dict = self._create_parameters_dict(PCA_ANNULAR_Params)

                all_params = {"algo_params": self, **rot_options}

                res = pca_annular(**all_params)

                self.cube_residuals = res[0]
                self.cube_residuals_der = res[1]
                if isinstance(res[2], list):
                    self.frames_final = res[2]
                else:
                    self.frame_final = res[2]

                if self.results is not None and self.frame_final is not None:
                    self.results.register_session(
                        params=params_dict,
                        frame=self.frame_final,
                        algo_name=self._algo_name[1],
                    )

            case Runmode.GRID:
                add_params = {
                    "full_output": full_output,
                    "verbose": verbose,
                }

                func_params = setup_parameters(
                    params_obj=self, fkt=pca_grid, **add_params
                )

                res = pca_grid(**func_params, **rot_options)

                if self.source_xy is not None and self.fwhm is not None:
                    (
                        self.cube_residuals,
                        self.frame_final,
                        self.dataframe,
                        self.opt_number_pc,
                    ) = res
                    if self.results is not None:
                        self.results.register_session(
                            params=func_params,
                            frame=self.frame_final,
                            algo_name=self._algo_name[2],
                        )
                elif self.full_output:
                    (
                        self.final_residuals_cube,
                        self.pc_list
                    ) = res
                else:
                    self.final_residuals_cube = res


            case Runmode.ANNULUS:
                add_params = {
                    "angs": self.angle_list,
                }

                func_params = setup_parameters(
                    params_obj=self, fkt=pca_annulus, **add_params
                )

                res = pca_annulus(**func_params, **rot_options)

                self.frame_final = res

                if self.results is not None:
                    self.results.register_session(
                        params=func_params,
                        frame=self.frame_final,
                        algo_name=self._algo_name[3],
                    )

            case _:
                raise ValueError("Invalid run mode selected.")

    def _find_pca_mode(self, res):
        """
        Identify the mode of PCA used and extracts return elements accordingly.

        Nine modes are currently known and each of them looks at specific
        conditions. Every mode and its set of conditions is verified to be True
        or not, and associates its return elements via the `match...case` if
        recognized.

        Parameters
        ----------
        res: any
            The return of the PCA function, can consist of a multitude of items.
        """
        conditions = {
            "cube": isinstance(self.cube, np.ndarray),
            "scale": self.scale_list is not None,
            "adimsdidouble": self.adimsdi == Adimsdi.DOUBLE,
            "adimsdisingle": self.adimsdi == Adimsdi.SINGLE,
            "ncompunit": isinstance(self.ncomp, (float, int)),
            "ncompit": isinstance(self.ncomp, (tuple, list)),
            "source": self.source_xy is not None,
            "nosource": self.source_xy is None,
            "reforsource": self.cube_ref is not None or self.source_xy is None,
            "nobatch": self.batch is None,
            "batch": self.batch is not None,
            "cubeorscale": isinstance(self.cube, str) or self.scale_list is None,
        }

        pca_modes = {
            ReturnList.ADIMSDI_DOUBLE: conditions["cube"]
            and conditions["scale"]
            and conditions["adimsdidouble"],
            ReturnList.ADIMSDI_SINGLE_NO_GRID: conditions["cube"]
            and conditions["scale"]
            and conditions["adimsdisingle"]
            and conditions["ncompunit"],
            ReturnList.ADIMSDI_SINGLE_GRID_NO_SOURCE: conditions["cube"]
            and conditions["scale"]
            and conditions["adimsdisingle"]
            and conditions["ncompit"]
            and conditions["nosource"],
            ReturnList.ADIMSDI_SINGLE_GRID_SOURCE: conditions["cube"]
            and conditions["scale"]
            and conditions["adimsdisingle"]
            and conditions["ncompit"]
            and conditions["source"],
            ReturnList.ADI_FULLFRAME_GRID: conditions["cubeorscale"]
            and conditions["reforsource"]
            and conditions["nobatch"]
            and conditions["ncompit"],
            ReturnList.ADI_INCREMENTAL_BATCH: conditions["cubeorscale"]
            and conditions["reforsource"]
            and conditions["batch"],
            ReturnList.ADI_FULLFRAME_STANDARD: conditions["cubeorscale"]
            and conditions["reforsource"]
            and conditions["nobatch"]
            and conditions["ncompunit"],
            ReturnList.PCA_GRID_SN: conditions["cubeorscale"]
            and conditions["source"]
            and conditions["ncompit"],
            ReturnList.PCA_ROT_THRESH: conditions["cubeorscale"]
            and conditions["source"]
            and conditions["ncompunit"],
        }

        pca_mode = None

        for mode, state in pca_modes.items():
            if state:
                pca_mode = mode
                break

        match (pca_mode):
            case ReturnList.ADIMSDI_DOUBLE:
                self.frame_final, self.cube_residuals, self.cube_residuals_der = res
            case ReturnList.ADIMSDI_SINGLE_NO_GRID:
                self.frame_final, self.cube_residuals, _ = res
            case ReturnList.ADIMSDI_SINGLE_GRID_NO_SOURCE:
                self.final_residuals_cube, self.frame_final, _ = res
            case ReturnList.ADIMSDI_SINGLE_GRID_SOURCE:
                self.final_residuals_cube, self.pc_list = res
            case ReturnList.ADI_FULLFRAME_GRID:
                if self.cube.ndim == 4:
                    self.frames_final, self.pc_list, _ = res
                else:
                    self.frames_final, self.pc_list = res
            case ReturnList.ADI_INCREMENTAL_BATCH:
                if self.cube.ndim == 4:
                    self.frame_final, self.pcs, self.medians, _ = res
                else:
                    self.frame_final, self.pcs, self.medians = res
            case ReturnList.ADI_FULLFRAME_STANDARD:
                if self.cube.ndim == 4:
                    (
                        self.frame_final,
                        self.pcs,
                        self.cube_reconstructed,
                        self.cube_residuals,
                        self.cube_residuals_der,
                        _,
                    ) = res
                else:
                    (
                        self.frame_final,
                        self.pcs,
                        self.cube_reconstructed,
                        self.cube_residuals,
                        self.cube_residuals_der,
                    ) = res
            case ReturnList.PCA_GRID_SN:
                if self.cube.ndim == 4:
                    self.final_residuals_cube, self.frame_final, _, self.opt_number_pc = res
                else:
                    self.final_residuals_cube, self.frame_final, _ = res
            case ReturnList.PCA_ROT_THRESH:
                if self.cube.ndim == 4:
                    (
                        self.frame_final,
                        self.cube_reconstructed,
                        self.cube_residuals,
                        self.cube_residuals_der,
                        _,
                    ) = res
                else:
                    (
                        self.frame_final,
                        self.cube_reconstructed,
                        self.cube_residuals,
                        self.cube_residuals_der,
                    ) = res
            case _:
                raise RuntimeError("No PCA mode could be identified.")


PCABuilder = dataclass_builder(PPPCA)
