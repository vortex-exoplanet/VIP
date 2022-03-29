
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from astropy.modeling import models
import hciplot
import vip_hci as vip
from vip_hci.preproc import (cube_recenter_2dfit, cube_recenter_dft_upsampling,
                             cube_recenter_satspots, frame_shift)
from sklearn.metrics import mean_squared_error

try:
    from IPython.core.display import display, HTML
    def html(s):
        display(HTML(s))
except:
    def html(s):
        print(s)

# import os
# print("VIP version: {} (from {})".format(vip.__version__,
#                                          os.path.dirname(vip.__file__)))


#                  888               888
#       o          888               888
#      d8b         888               888
#     d888b        88888b.   .d88b.  888 88888b.   .d88b.  888d888
# "Y888888888P"    888 "88b d8P  Y8b 888 888 "88b d8P  Y8b 888P"
#   "Y88888P"      888  888 88888888 888 888  888 88888888 888
#   d88P"Y88b      888  888 Y8b.     888 888 d88P Y8b.     888
#  dP"     "Yb     888  888  "Y8888  888 88888P"   "Y8888  888
#                                        888
#                                        888
#                                        888

seed = np.random.RandomState(42)


def resource(*args):
    try:
        import os
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), *args)
    except Exception:  # __file__ is not available
        return os.path.join(*args)


def shift_cube(cube, randax, randay):
    return np.array([frame_shift(cube[i], randay[i], randax[i])
                     for i in range(cube.shape[0])])


def create_cube_with_gauss2d(shape=(4, 9, 9), mean=4, stddev=1):
    nframes, sizex, sizey = shape

    try:
        x_mean, y_mean = mean
    except Exception:
        x_mean = y_mean = mean

    gauss = models.Gaussian2D(amplitude=1, x_mean=x_mean, y_mean=y_mean,
                              x_stddev=stddev, y_stddev=stddev)
    x = np.arange(sizex)
    y = np.arange(sizey)
    x, y = np.meshgrid(x, y)
    gaus_im = gauss(x, y)
    return np.array([gaus_im for _ in range(nframes)])


def create_cube_with_gauss2d_ring(stddev_inner, stddev_outer, **kwargs):
    outer = create_cube_with_gauss2d(stddev=stddev_outer, **kwargs)
    inner = create_cube_with_gauss2d(stddev=stddev_inner, **kwargs)
    return outer - inner


def create_cube_with_satspots(n_frames=6, wh=31, star_fwhm=3, debug=False):
    global seed
    shape = (n_frames, wh, wh)
    star = create_cube_with_gauss2d(shape=shape, mean=wh // 2, stddev=star_fwhm)

    # make sure satspot is neither too close to star nor at the edge of the
    # image
    diagonal = seed.uniform(4 * star_fwhm, wh // 2)
    d = diagonal / np.sqrt(2)

    sat1_coords = (wh // 2 - d, wh // 2 + d)
    sat2_coords = (wh // 2 + d, wh // 2 + d)
    sat3_coords = (wh // 2 - d, wh // 2 - d)
    sat4_coords = (wh // 2 + d, wh // 2 - d)

    sat1 = create_cube_with_gauss2d(shape=shape, mean=sat1_coords, stddev=1)
    sat2 = create_cube_with_gauss2d(shape=shape, mean=sat2_coords, stddev=1)
    sat3 = create_cube_with_gauss2d(shape=shape, mean=sat3_coords, stddev=1)
    sat4 = create_cube_with_gauss2d(shape=shape, mean=sat4_coords, stddev=1)

    cube = star + sat1 + sat2 + sat3 + sat4

    if debug:
        hciplot.plot_frames(cube[0])

    return cube, [sat1_coords, sat2_coords, sat3_coords, sat4_coords]


def do_recenter(method, cube, shiftx, shifty, errormsg, mse=1e-2,
                mse_skip_first=False, n_frames=6, debug=False, **kwargs):
    #===== shift cube
    shifted_cube = shift_cube(cube, shiftx, shifty)

    if debug:
        html("<h3>===== {}({}) =====</h3>".format(
            method.__name__,
            ", ".join("{}={}".format(k, v) for k, v in kwargs.items())))

    #===== recentering
    rec_res = method(shifted_cube, debug=debug, **kwargs)
    
    recentered_cube= rec_res[0]
    unshifty= rec_res[1]
    unshiftx= rec_res[2]

    if debug:
        hciplot.plot_frames(cube, title="input cube")
        hciplot.plot_frames(shifted_cube, title="shifted cube")
        hciplot.plot_frames(recentered_cube, title="recentered cube")

    if debug:
        hciplot.plot_frames(cube[1], recentered_cube[1], shifted_cube[1],
                            label=["cube[1]", "recentered[1]", "shifted[1]"])

    if mse_skip_first:
        if debug:
            print("\033[33mfirst shift ignored for MSE\033[0m")
        shiftx = shiftx[1:]
        shifty = shifty[1:]
        unshiftx = unshiftx[1:]
        unshifty = unshifty[1:]

    if debug:
        try:
            import pandas as pd
            from IPython.display import display
            p = pd.DataFrame(np.array([
                shiftx,
                -unshiftx,
                shifty,
                -unshifty
            ]).T, columns=["x", "un-x", "y", "un-y"])
            print("\033[33mshifts:\033[0m")
            display(p)
        except:
            print("\033[33mcalculated shifts:\033[0m", unshiftx)
            print(" " * 18, unshifty)
            print("\033[33moriginal shifts\033[0m:  ", -shiftx)
            print(" " * 18, -shifty)
        print("\033[33merrors:\033[0m", mean_squared_error(
            shiftx, -unshiftx), mean_squared_error(shifty, -unshifty))

    #===== verify error
    assert mean_squared_error(shiftx, -unshiftx) < mse, errormsg
    assert mean_squared_error(shifty, -unshifty) < mse, errormsg

    if debug:
        print("\033[32mpassed.\033[0m")


#                  888                      888
#       o          888                      888
#      d8b         888                      888
#     d888b        888888  .d88b.  .d8888b  888888 .d8888b
# "Y888888888P"    888    d8P  Y8b 88K      888    88K
#   "Y88888P"      888    88888888 "Y8888b. 888    "Y8888b.
#   d88P"Y88b      Y88b.  Y8b.          X88 Y88b.       X88
#  dP"     "Yb      "Y888  "Y8888   88888P'  "Y888  88888P'


def test_2d(debug=False):
    """
    tests `cube_recenter_2dfit`. The data cube is generated from a 2D gaussian 
    (positive case) or a 2D "ring" (difference of two 2D gaussians, negative 
    case).
    """
    global seed

    if debug:
        html("<h2>===== test_2d =====</h2>")

    method = cube_recenter_2dfit
    errormsg = 'Error when recentering with 2d {} fitting method'
    n_frames = 6

    randax = seed.uniform(-1, 1, size=n_frames)
    randay = seed.uniform(-1, 1, size=n_frames)

    for model, name in {"moff": "Moffat", "gauss": "Gaussian"}.items():

        #===== odd
        cube = create_cube_with_gauss2d(shape=(n_frames, 9, 9), mean=4,
                                        stddev=1)

        method_args = dict(fwhm=1, subi_size=5, model=model, verbose=False,
                           negative=False, full_output=True, plot=False)
        do_recenter(method, cube, randax, randay,
                    errormsg=errormsg.format(name), debug=debug, **method_args)

        #===== even
        cube = create_cube_with_gauss2d(shape=(n_frames, 10, 10),mean=5,
                                        stddev=1)

        method_args = dict(fwhm=1, subi_size=6, model=model, verbose=False,
                           negative=False, full_output=True, plot=False)
        do_recenter(method, cube, randax, randay,
                    errormsg=errormsg.format(name), debug=debug, **method_args)

        #===== odd negative (ring)
        cube = create_cube_with_gauss2d_ring(shape=(n_frames, 9, 9), mean=4,
                                             stddev_outer=3, stddev_inner=2)

        method_args = dict(fwhm=1, subi_size=5, model=model, verbose=False,
                           negative=True, full_output=True, plot=False)
        do_recenter(method, cube, randax, randay,
                    errormsg=errormsg.format(name), debug=debug, **method_args)

        #===== even negative (ring)
        cube = create_cube_with_gauss2d_ring(shape=(n_frames, 10, 10), mean=5,
                                             stddev_outer=3, stddev_inner=2)

        method_args = dict(fwhm=1, subi_size=6, model=model, verbose=False,
                           negative=True, full_output=True, plot=False)
        do_recenter(method, cube, randax, randay,
                    errormsg=errormsg.format(name), debug=debug, **method_args)


def test_dft(debug=False):
    global seed
    if debug:
        html("<h2>===== test_dft =====</h2>")

    method = cube_recenter_dft_upsampling
    method_args_additional = dict(verbose=True, full_output=True, plot=False)
    errormsg = 'Error when recentering with DFT upsampling method'
    n_frames = 6

    shift_magnitude = 2
    randax = seed.uniform(-shift_magnitude, shift_magnitude, size=n_frames)
    randay = seed.uniform(-shift_magnitude, shift_magnitude, size=n_frames)
    randax[0] = 0  # do not shift first frame
    randay[0] = 0

    #===== odd, subi_size=None
    size = 9
    mean = size // 2
    cube = create_cube_with_gauss2d(shape=(n_frames, size, size), mean=mean,
                                    stddev=1)

    method_args = dict(center_fr1=(mean,mean), subi_size=None, negative=False,
                       **method_args_additional)
    do_recenter(method, cube, randax, randay, errormsg=errormsg,
                mse_skip_first=True, debug=debug, **method_args)

    #===== even, subi_size
    size = 10
    mean = size // 2 #- 0.5 # 0-indexed
    cube = create_cube_with_gauss2d(shape=(n_frames, size, size), mean=mean,
                                    stddev=1)

    method_args = dict(center_fr1=(mean,mean), subi_size=8, negative=False,
                       **method_args_additional)
    do_recenter(method, cube, randax, randay, errormsg=errormsg,
                mse_skip_first=True, debug=debug, **method_args)

    #===== odd negative (ring), subi_size
    size = 15
    mean = size // 2
    cube = create_cube_with_gauss2d_ring(shape=(n_frames, size, size),
                                         mean=mean, stddev_outer=3,
                                         stddev_inner=2)

    method_args = dict(center_fr1=(mean,mean), subi_size=12, negative=True,
                       **method_args_additional)
    do_recenter(method, cube, randax, randay, errormsg=errormsg,
                mse_skip_first=True, debug=debug, **method_args)

    #===== even negative (ring), subi_size=None
    size = 16
    mean = size // 2 #- 0.5
    cube = create_cube_with_gauss2d_ring(shape=(n_frames, size, size),
                                         mean=mean, stddev_outer=3,
                                         stddev_inner=2)

    method_args = dict(center_fr1=(mean,mean), subi_size=None, negative=True,
                       **method_args_additional)
    do_recenter(method, cube, randax, randay, errormsg=errormsg,
                mse_skip_first=True, debug=debug, **method_args)


def test_dft_image(debug=False):
    """
    notes:
    ======
    don't forget to specify `mse_skip_first`, as the shift of the first frame 
    does not make sense (provided by the user / determined by *gaussian* fit)
    """
    global seed
    if debug:
        html("<h2>===== test_dft_image =====</h2>")

    method = cube_recenter_dft_upsampling
    errormsg = 'Error when recentering with DFT upsampling method'
    n_frames = 6

    #===== datacube
    img = vip.fits.open_fits(resource('naco_betapic_single.fits'))
    cube = np.array([img, ] * n_frames)

    #===== shift
    randax = seed.uniform(-1, 1, size=n_frames)
    randay = seed.uniform(-1, 1, size=n_frames)
    randax[0] = 0  # do not shift first frame
    randay[0] = 0

    #===== recenter
    method_args = dict(center_fr1=(51,51), subi_size=None, verbose=True,
                       negative=True, full_output=True, plot=False)
    do_recenter(method, cube, randax, randay, errormsg=errormsg,
                mse_skip_first=True, debug=debug, **method_args)


def test_satspots_image(debug=False):
    global seed
    if debug:
        html("<h2>===== test_satspots_image =====</h2>")

    method = cube_recenter_satspots
    errormsg = 'Error when recentering with satellite spots'
    n_frames = 6

    #===== datacube
    img = vip.fits.open_fits(resource('SPHERE_satspots_centered.fits'))
    cube = np.array([img, ] * n_frames)

    #===== shift
    randax = seed.uniform(-1, 1, size=n_frames)
    randay = seed.uniform(-1, 1, size=n_frames)

    #===== recenter
    spotcoords = [(41, 109), (109, 109), (41, 41), (109, 41)]  # NW NE SW SE
    method_args = dict(xy=spotcoords, subi_size=25, plot=False,
                       full_output=True, verbose=False)
    do_recenter(method, cube, randax, randay, errormsg=errormsg, debug=debug,
                **method_args)


def test_satspots(debug=False):
    global seed
    if debug:
        html("<h2>===== test_satspots =====</h2>")

    method = cube_recenter_satspots
    errormsg = 'Error when recentering with satellite spots'
    n_frames = 2

    #===== datacube
    cube, spotcoords = create_cube_with_satspots(n_frames=n_frames)

    #===== shift
    shift_magnitude = 1
    randax = seed.uniform(-shift_magnitude, 0, size=n_frames)
    randay = seed.uniform(0, shift_magnitude, size=n_frames)

    #===== recenter
    method_args = dict(xy=spotcoords, subi_size=9, plot=False,
                       full_output=True, verbose=False)
    do_recenter(method, cube, randax, randay, errormsg=errormsg, debug=debug,
                **method_args)

