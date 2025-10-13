#! /usr/bin/env python
"""Functions useful for disk model interpolation for requested parameters\
falling within the provided grid."""

__author__ = 'Valentin Christiaens'
__all__ = ['interpolate_model']

import numpy as np
from scipy.ndimage import map_coordinates
from .utils_negfc import find_nearest


def interpolate_model(params, grid_param_list, model_grid, interp_order=-1,
                      multispectral=False, verbose=False):
    """Interpolate model grid for requested parameters.

    Parameters
    ----------
    params : tuple
        Set of models parameters for which the model grid has to be
        interpolated.
    grid_param_list : list of 1d numpy arrays/lists
        List/numpy 1d arrays with available grid of model parameters (should
        only contain the sampled parameters, not the models themselves).
    model_grid : numpy N-d array, optional
        Grid of model spectra for each free parameter of the given grid. For
        a single (resp. multi) wavelength model, the model grid should have N+2
        (resp. N+3) dimensions, where N is the number of free parameters in the
        grid (i.e. the length of grid_param_list).
    interp_order: int or tuple of int, optional, {-1,0,1}
        Interpolation mode for model interpolation. If a tuple of integers, the
        length should match the number of grid dimensions and will trigger a
        different interpolation mode for the different parameters.
            - -1: Order 1 spline interpolation in logspace for the parameter
            - 0: nearest neighbour model
            - 1: Order 1 spline interpolation

    multispectral: bool, optional
        Whether the model grid is computed for various wavelenghts - e.g. for
        IFS data. In this case, the wavelength dimension should be the third to
        last in the input model_grid.
    verbose: bool, optional
        Whether to print more information during the interpolation.

    Returns
    -------
    model : 2d or 3d numpy array
        Interpolated model for input parameters. First column corresponds
        to wavelengths, and the second contains model values.

    """

    def _den_to_bin(denary, ndigits=3):
        """Convert denary to binary number, keeping n digits for binary."""
        binary = ""
        while denary > 0:
            # A left shift in binary means /2
            binary = str(denary % 2) + binary
            denary = denary//2
        if len(binary) < ndigits:
            pad = '0'*(ndigits-len(binary))
        else:
            pad = ''
        return pad+binary

    n_params_tot = len(grid_param_list)

    if isinstance(interp_order, (int, bool)):
        interp_order = [interp_order]*n_params_tot
        interp_order = tuple(interp_order)

    if np.sum(np.abs(interp_order)) == 0:
        idx_tmp = []
        for nn in range(n_params_tot):
            idx_tmp.append(find_nearest(grid_param_list[nn], params[nn],
                                        output='index'))
        idx_tmp = tuple(idx_tmp)
        return model_grid[idx_tmp]

    else:
        if len(interp_order) != n_params_tot:
            msg = "if a tuple, interp_order should have same length as the "
            msg += "number of grid dimensions"
            raise TypeError(msg)
        else:
            for i in range(n_params_tot):
                if interp_order[i] not in [-1, 0, 1]:
                    msg = "interp_order values should be -1, 0, or 1"
                    raise TypeError(msg)

        # multispectral or not?
        if multispectral:
            ndim = 3
        else:
            ndim = 2

        # first compute new subgrid "coords" for interpolation
        if verbose:
            print("Computing new coords for interpolation")
        constr = ['floor=', 'ceil=']
        new_coords = np.zeros([n_params_tot, 1])
        sub_grid_param = np.zeros([n_params_tot, 2])
        for nn in range(n_params_tot):
            grid_tmp = grid_param_list[nn]
            params_tmp = params[nn]
            for ii in range(2):
                sub_grid_param[nn, ii] = find_nearest(grid_tmp,
                                                      params_tmp,
                                                      constraint=constr[ii],
                                                      output='value')
            if interp_order[nn] == -1:
                num = np.log(params_tmp/sub_grid_param[nn, 0])
                denom = np.log(sub_grid_param[nn, 1]/sub_grid_param[nn, 0])
            else:
                num = (params_tmp-sub_grid_param[nn, 0])
                denom = (sub_grid_param[nn, 1]-sub_grid_param[nn, 0])
            new_coords[nn, 0] = num/denom
            if interp_order[nn] == 0:
                new_coords[nn, 0] = round(new_coords[nn, 0])
            # if interp_order == -1:
            #     # consider it in log space
            #     num = np.log(params_tmp/sub_grid_param[nn, 0])
            #     denom = np.log(sub_grid_param[nn, 1]/sub_grid_param[nn, 0])
            # else:
            #     num = params_tmp-sub_grid_param[nn, 0]
            #     denom = sub_grid_param[nn, 1]-sub_grid_param[nn, 0]
            # new_coords[nn, 0] = num/denom

        # make subgrid in the model grid
        if verbose:
            print("Making sub-grid of models")
        subgrid = []
        subgrid_idx = np.zeros([n_params_tot, 2], dtype=np.int32)
        for nn in range(n_params_tot):
            grid_tmp = grid_param_list[nn]
            params_tmp = params[nn]
            for ii in range(2):
                subgrid_idx[nn, ii] = find_nearest(grid_tmp, params_tmp,
                                                   constraint=constr[ii],
                                                   output='index')
        for dd in range(2**n_params_tot):
            str_indices = _den_to_bin(dd, n_params_tot)
            idx_tmp = []
            for nn in range(n_params_tot):
                idx_tmp.append(subgrid_idx[nn, int(str_indices[nn])])
            subgrid.append(model_grid[tuple(idx_tmp)])
        # reshape grid
        subgrid = np.array(subgrid)
        dims = [2]*n_params_tot
        dims += [model_grid.shape[-ndim+i] for i in range(ndim)]
        dims = tuple(dims)
        subgrid = subgrid.reshape(dims)

        # make last dimensions (model images) come first
        if multispectral:
            subgrid = np.moveaxis(subgrid, [-3, -2, -1], [0, 1, 2])
        else:
            subgrid = np.moveaxis(subgrid, [-2, -1], [0, 1])

        # interpolate in the subgrid
        model = np.zeros(model_grid.shape[-ndim:])

        if multispectral:
            nz, ny, nx = model_grid.shape[-ndim:]
            for zz in range(nz):
                for yy in range(ny):
                    for xx in range(nx):
                        model[zz, yy, xx] = map_coordinates(subgrid[zz, yy, xx],
                                                            new_coords,
                                                            order=1)
        else:
            ny, nx = model_grid.shape[-ndim:]
            for yy in range(ny):
                for xx in range(nx):
                    model[yy, xx] = map_coordinates(subgrid[yy, xx],
                                                    new_coords,
                                                    order=1)

        return model
