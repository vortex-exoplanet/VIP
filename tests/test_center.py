from __future__ import division, print_function

import numpy as np
from astropy.modeling.models import Gaussian2D
from vip_hci.preproc import cube_recenter_2dfit, cube_recenter_dft_upsampling, frame_shift
from sklearn.metrics import mean_squared_error
import pytest


def test_cube_with_gauss2d():
    def create_cube_with_gauss2d(shape=(4, 9, 9), x_mean=4., y_mean=4,
                                 x_stddev=1, y_stddev=1):
        nframes, sizex, sizey = shape
        gauss = Gaussian2D(amplitude=1, x_mean=x_mean, y_mean=y_mean,
                           x_stddev=x_stddev, y_stddev=y_stddev)
        x = np.arange(sizex)
        y = np.arange(sizey)
        x, y = np.meshgrid(x, y)
        gaus_im = gauss(x, y)
        return np.array([gaus_im for i in range(nframes)])
    ############################################################################

    n_frames = 6
    megerrg = 'Error when recentering with Gaussian fitting'
    msgerrm = 'Error when recentering with Moffat fitting'
    megerrd = 'Error when recentering with DFT upsampling'

    ### odd case ###
    cube1 = create_cube_with_gauss2d(shape=(n_frames, 9, 9), x_mean=4, y_mean=4,
                                     x_stddev=1, y_stddev=1)
    randax1 = np.random.uniform(-1, 1, size=n_frames)
    randay1 = np.random.uniform(-1, 1, size=n_frames)
    cubeshi1 = np.array([frame_shift(cube1[i], randay1[i], randax1[i]) for i in
                         range(n_frames)])
    # Gaussian fitting
    _, shiyg, shixg = cube_recenter_2dfit(cubeshi1, fwhm=1, subi_size=5,
                                          model='gauss', verbose=False,
                                          full_output=True)
    assert mean_squared_error(randax1, -shixg) < 1e-2, megerrg
    assert mean_squared_error(randay1, -shiyg) < 1e-2, megerrg
    # Moffat fitting
    _, shiym, shixm = cube_recenter_2dfit(cubeshi1, fwhm=1, subi_size=5,
                                          model='moff', verbose=False,
                                          full_output=True)
    assert mean_squared_error(randax1, -shixm) < 1e-2, msgerrm
    assert mean_squared_error(randay1, -shiym) < 1e-2, msgerrm
    # DFT upsampling
    _, shiyd, shixd = cube_recenter_dft_upsampling(cubeshi1, 4, 4, subi_size=5,
                                                   verbose=False,
                                                   full_output=True)
    assert mean_squared_error(randax1, -shixd) < 1e-2, megerrd
    assert mean_squared_error(randay1, -shiyd) < 1e-2, megerrd

    ### even case ###
    cube2 = create_cube_with_gauss2d(shape=(n_frames, 10, 10), x_mean=4.5,
                                     y_mean=4.5, x_stddev=1, y_stddev=1)
    randax2 = np.random.uniform(-1, 1, size=n_frames)
    randay2 = np.random.uniform(-1, 1, size=n_frames)
    cubeshi2 = np.array([frame_shift(cube2[i], randay2[i], randax2[i]) for i in
                         range(n_frames)])
    # Gaussian fitting
    _, shiyg, shixg = cube_recenter_2dfit(cubeshi2, fwhm=1, subi_size=6,
                                          model='gauss', verbose=False,
                                          full_output=True)
    assert mean_squared_error(randax2, -shixg) < 1e-2, megerrg
    assert mean_squared_error(randay2, -shiyg) < 1e-2, megerrg
    # Moffat fitting
    _, shiym, shixm = cube_recenter_2dfit(cubeshi2, fwhm=1, subi_size=6,
                                          model='moff', verbose=False,
                                          full_output=True)
    assert mean_squared_error(randax2, -shixm) < 1e-2, msgerrm
    assert mean_squared_error(randay2, -shiym) < 1e-2, msgerrm
    # DFT upsampling
    _, shiyd, shixd = cube_recenter_dft_upsampling(cubeshi2, 4, 4, subi_size=6,
                                                   verbose=False,
                                                   full_output=True)
    assert mean_squared_error(randax2, -shixd) < 1e-2, megerrd
    assert mean_squared_error(randay2, -shiyd) < 1e-2, megerrd
