#! /usr/bin/env python

"""
Module containing functions for cubes frame registration.
"""

from __future__ import division, print_function

__author__ = 'Carlos Alberto Gomez Gonzalez, V. Christiaens @ ULg/UChile, '\
             'G. Ruane'
__all__ = ['frame_shift',
           'frame_center_radon',
           'frame_center_satspots',
           'cube_recenter_satspots',
           'cube_recenter_radon',
           'cube_recenter_dft_upsampling',
           'cube_recenter_2dfit',
           'cube_recenter_via_speckles']

import numpy as np
import warnings
import itertools as itt

try:
    import cv2
    no_opencv = False
except ImportError:
    msg = "Opencv python bindings are missing."
    warnings.warn(msg, ImportWarning)
    no_opencv = True

from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from skimage.transform import radon
from skimage.feature import register_translation
from multiprocessing import Pool, cpu_count
from matplotlib import pyplot as plt
from . import frame_crop
from ..conf import time_ini, timing, Progressbar
from ..conf.utils_conf import vip_figsize
from ..conf.utils_conf import eval_func_tuple as EFT
from ..var import (get_square, frame_center, get_annulus, pp_subplots,
                   fit_2dmoffat, fit_2dgaussian, cube_filter_lowpass,
                   cube_filter_highpass)
from ..preproc import cube_crop_frames


def frame_shift(array, shift_y, shift_x, imlib='opencv',
                interpolation='lanczos4', border_mode='reflect'):
    """ Shifts a 2D array by shift_y, shift_x. Boundaries are filled with zeros.

    Parameters
    ----------
    array : array_like
        Input 2d array.
    shift_y, shift_x: float
        Shifts in y and x directions.
    imlib : {'opencv', 'ndimage-fourier', 'ndimage-interp'}, string optional
        Library or method used for performing the image shift.
        'ndimage-fourier', does a fourier shift operation and preserves better
        the pixel values (therefore the flux and photometry). Interpolation
        based shift ('opencv' and 'ndimage-interp') is faster than the fourier
        shift. 'opencv' is recommended when speed is critical.
    interpolation : str, optional
        Only used in case of imlib is set to 'opencv' or 'ndimage-interp'
        (Scipy.ndimage), where the images are shifted via interpolation.
        For Scipy.ndimage the options are: 'nearneig', bilinear', 'bicuadratic',
        'bicubic', 'biquartic' or 'biquintic'. The 'nearneig' interpolation is
        the fastest and the 'biquintic' the slowest. The 'nearneig' is the
        poorer option for interpolation of noisy astronomical images.
        For Opencv the options are: 'nearneig', 'bilinear', 'bicubic' or
        'lanczos4'. The 'nearneig' interpolation is the fastest and the
        'lanczos4' the slowest and accurate. 'lanczos4' is the default for
        Opencv and 'biquartic' for Scipy.ndimage.
    border_mode : {'reflect', 'nearest', 'constant', 'mirror', 'wrap'}
        Points outside the boundaries of the input are filled accordingly.
        With 'reflect', the input is extended by reflecting about the edge of
        the last pixel. With 'nearest', the input is extended by replicating the
        last pixel. With 'constant', the input is extended by filling all values
        beyond the edge with zeros. With 'mirror', the input is extended by
        reflecting about the center of the last pixel. With 'wrap', the input is
        extended by wrapping around to the opposite edge. Default is 'reflect'.
    
    Returns
    -------
    array_shifted : array_like
        Shifted 2d array.

    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')
    
    image = array.copy()

    if imlib == 'ndimage-fourier':
        shift_val = (shift_y, shift_x)
        array_shifted = fourier_shift(np.fft.fftn(image), shift_val)
        array_shifted = np.fft.ifftn(array_shifted)
        array_shifted = array_shifted.real

    elif imlib == 'ndimage-interp':
        if interpolation == 'nearneig':
            order = 0
        elif interpolation == 'bilinear':
            order = 1
        elif interpolation == 'bicuadratic':
            order = 2
        elif interpolation == 'bicubic':
            order = 3
        elif interpolation == 'biquartic' or interpolation == 'lanczos4':
            order = 4
        elif interpolation == 'biquintic':
            order = 5
        else:
            raise ValueError('Scipy.ndimage interpolation method not recognized')

        if border_mode not in ['reflect', 'nearest', 'constant', 'mirror',
                               'wrap']:
            raise ValueError('`border_mode` not recognized')

        array_shifted = shift(image, (shift_y, shift_x), order=order,
                              mode=border_mode)
    
    elif imlib == 'opencv':
        if no_opencv:
            msg = 'Opencv python bindings cannot be imported. Install opencv or'
            msg += ' set imlib to ndimage-fourier or ndimage-interp'
            raise RuntimeError(msg)

        if interpolation == 'bilinear':
            intp = cv2.INTER_LINEAR
        elif interpolation == 'bicubic':
            intp= cv2.INTER_CUBIC
        elif interpolation == 'nearneig':
            intp = cv2.INTER_NEAREST
        elif interpolation == 'lanczos4':
            intp = cv2.INTER_LANCZOS4
        else:
            raise ValueError('Opencv interpolation method not recognized')

        if border_mode == 'mirror':
            bormo = cv2.BORDER_REFLECT_101  # gfedcb|abcdefgh|gfedcba
        elif border_mode == 'reflect':
            bormo = cv2.BORDER_REFLECT  # fedcba|abcdefgh|hgfedcb
        elif border_mode == 'wrap':
            bormo = cv2.BORDER_WRAP  # cdefgh|abcdefgh|abcdefg
        elif border_mode == 'constant':
            bormo = cv2.BORDER_CONSTANT  # iiiiii|abcdefgh|iiiiiii
        elif border_mode == 'nearest':
            bormo = cv2.BORDER_REPLICATE  # aaaaaa|abcdefgh|hhhhhhh
        else:
            raise ValueError('`border_mode` not recognized')

        image = np.float32(image)
        y, x = image.shape
        M = np.float32([[1,0,shift_x],[0,1,shift_y]])
        array_shifted = cv2.warpAffine(image, M, (x,y), flags=intp,
                                       borderMode=bormo)

    else:
        raise ValueError('Image transformation library not recognized')
    
    return array_shifted


def frame_center_satspots(array, xy, subi_size=19, sigfactor=6, shift=False,
                          imlib='opencv', interpolation='lanczos4',
                          debug=False, verbose=True):
    """ Finds the center of a frame with waffle/satellite spots (e.g. for 
    VLT/SPHERE). The method used to determine the center is by centroiding the
    4 spots via a 2d Gaussian fit and finding the intersection of the 
    lines they create (see Notes). This method is very sensitive to the SNR of 
    the satellite spots, therefore thresholding of the background pixels is 
    performed. If the results are too extreme, the debug parameter will allow to 
    see in depth what is going on with the fit (maybe you'll need to adjust the 
    sigfactor for the background pixels thresholding).  
    
    Parameters
    ----------
    array : array_like, 2d
        Image or frame.
    xy : tuple
        Tuple with coordinates X,Y of the 4 satellite spots. When the spots are
        in an X configuration, the order is the following: top-left, top-right,
        bottom-left and bottom-right. When the spots are in an + (cross-like)
        configuration, the order is the following: top, right, left, bottom.
    subi_size : int, optional
        Size of subimage where the fitting is done.
    sigfactor : int, optional
        The background pixels will be thresholded before fitting a 2d Gaussian
        to the data using sigma clipped statistics. All values smaller than
        (MEDIAN + sigfactor*STDDEV) will be replaced by small random Gaussian 
        noise. 
    shift : bool, optional
        If True the image is shifted.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    debug : bool, optional
        If True debug information is printed and plotted.
    verbose : bool, optional
        If True the intersection and shifts information is printed out.
    
    Returns
    -------
    shifty, shiftx 
        Shift Y,X to get to the true center.
    If shift is True then the shifted image is returned along with the shifts.
    
    Notes
    -----
    linear system:
    A1 * x + B1 * y = C1
    A2 * x + B2 * y = C2
    
    Cramer's rule - solution can be found in determinants:
    x = Dx/D
    y = Dy/D
    where D is main determinant of the system:  A1 B1
                                                A2 B2
    and Dx and Dy can be found from matrices:  C1 B1
                                               C2 B2
    and  A1 C1
         A2 C2
    C column consequently substitutes the coef. columns of x and y

    L stores our coefs A, B, C of the line equations.
    For D: L1[0] L1[1]   for Dx: L1[2] L1[1]   for Dy: L1[0] L1[2]
           L2[0] L2[1]           L2[2] L2[1]           L2[0] L2[2]
    """
    def line(p1, p2):
        """ produces coefs A, B, C of line equation by 2 points
        """
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])
        return A, B, -C

    def intersection(L1, L2):
        """ finds intersection point (if any) of 2 lines provided by coefs
        """
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return None
        
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')
    if len(xy) != 4:
        raise TypeError('Input waffle spot coordinates in wrong format')
    
    cy, cx = frame_center(array)
    
    # Top left
    si1, y1, x1 = get_square(array, subi_size, xy[0][1], xy[0][0],
                             position=True)
    cent2dgy_1, cent2dgx_1 = fit_2dgaussian(si1, theta=135, crop=False, 
                                            threshold=True, sigfactor=sigfactor, 
                                            debug=debug)
    cent2dgx_1 += x1
    cent2dgy_1 += y1
    # Top right
    si2, y2, x2 = get_square(array, subi_size, xy[1][1], xy[1][0],
                             position=True)
    cent2dgy_2, cent2dgx_2 = fit_2dgaussian(si2, theta=45, crop=False, 
                                            threshold=True, sigfactor=sigfactor, 
                                            debug=debug)
    cent2dgx_2 += x2
    cent2dgy_2 += y2 
    #  Bottom left
    si3, y3, x3 = get_square(array, subi_size, xy[2][1], xy[2][0],
                             position=True)
    cent2dgy_3, cent2dgx_3 = fit_2dgaussian(si3, theta=45, crop=False, 
                                            threshold=True, sigfactor=sigfactor, 
                                            debug=debug)
    cent2dgx_3 += x3
    cent2dgy_3 += y3
    #  Bottom right
    si4, y4, x4 = get_square(array, subi_size, xy[3][1], xy[3][0],
                             position=True)
    cent2dgy_4, cent2dgx_4 = fit_2dgaussian(si4, theta=135, crop=False, 
                                            threshold=True, sigfactor=sigfactor, 
                                            debug=debug)
    cent2dgx_4 += x4
    cent2dgy_4 += y4
    
    if debug: 
        pp_subplots(si1, si2, si3, si4, colorb=True)
        print('Centroids X,Y:')
        print(cent2dgx_1, cent2dgy_1)
        print(cent2dgx_2, cent2dgy_2)
        print(cent2dgx_3, cent2dgy_3)
        print(cent2dgx_4, cent2dgy_4)

    L1 = line([cent2dgx_1, cent2dgy_1], [cent2dgx_4, cent2dgy_4])
    L2 = line([cent2dgx_2, cent2dgy_2], [cent2dgx_3, cent2dgy_3])
    R = intersection(L1, L2)

    msgerr = "Check that the order of the tuples in `xy` is correct and"
    msgerr += " the satellite spots have good S/N"
    if R is not None:
        shiftx = cx - R[0]
        shifty = cy - R[1]

        if np.abs(shiftx) < cx * 2 and np.abs(shifty) < cy * 2:
            if debug or verbose:
                print('Intersection coordinates (X,Y):', R[0], R[1], '\n')
                print('Shifts (X,Y): {:.3f}, {:.3f}'.format(shiftx, shifty))

            if shift:
                array_rec = frame_shift(array, shifty, shiftx, imlib=imlib,
                                        interpolation=interpolation)
                return array_rec, shifty, shiftx
            else:
                return shifty, shiftx
        else:
            raise RuntimeError("Too large shifts. " + msgerr)
    else:
        raise RuntimeError("Something went wrong, no intersection found. " +
                           msgerr)


def cube_recenter_satspots(array, xy, subi_size=19, sigfactor=6, plot=True,
                           debug=False, verbose=True):
    """ Function analog to frame_center_satspots but for image sequences. It 
    actually will call frame_center_satspots for each image in the cube. The
    function also returns the shifted images (not recommended to use when the 
    shifts are of a few percents of a pixel) and plots the histogram of the
    shifts and calculate its statistics. This is important to assess the 
    dispersion of the star center by using artificial waffle/satellite spots
    (like those in VLT/SPHERE images) and evaluate the uncertainty of the 
    position of the center. The use of the shifted images is not recommended.   
    
    Parameters
    ----------
    array : array_like, 3d
        Input cube.
    xy : tuple
        Tuple with coordinates X,Y of the 4 satellite spots. When the spots are
        in an X configuration, the order is the following: top-left, top-right,
        bottom-left and bottom-right. When the spots are in an + (cross-like)
        configuration, the order is the following: top, right, left, bottom.
    subi_size : int, optional
        Size of subimage where the fitting is done.
    sigfactor : int, optional
        The background pixels will be thresholded before fitting a 2d Gaussian
        to the data using sigma clipped statistics. All values smaller than
        (MEDIAN + sigfactor*STDDEV) will be replaced by small random Gaussian 
        noise.
    plot : bool, optional
        Whether to plot the shifts.
    debug : bool, optional
        If True debug information is printed and plotted (fit and residuals,
        intersections and shifts). This has to be used carefully as it can 
        produce too much output and plots.
    verbose : bool, optional
        Whether to print to stdout the timing and aditional info.
    
    Returns
    ------- 
    array_rec
        The shifted cube.
    shift_y, shift_x
        Shifts Y,X to get to the true center for each image.
    
    """    
    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array')

    if verbose:
        start_time = time_ini()

    n_frames = array.shape[0]
    shift_x = np.zeros((n_frames))
    shift_y = np.zeros((n_frames))
    array_rec = []
    
    if verbose:
        print('Looping through the frames, fitting the intersections:')
    for i in Progressbar(range(n_frames), verbose=verbose):
        res = frame_center_satspots(array[i], xy, debug=debug, shift=True,
                                    subi_size=subi_size, sigfactor=sigfactor,
                                    verbose=False)
        array_rec.append(res[0])
        shift_y[i] = res[1] 
        shift_x[i] = res[2]          
       
    if verbose:
        timing(start_time)

    if plot:
        plt.figure(figsize=vip_figsize)
        plt.plot(shift_x, 'o-', label='Shifts in x', alpha=0.5)
        plt.plot(shift_y, 'o-', label='Shifts in y', alpha=0.5)
        plt.legend(loc='best')
        plt.grid('on', alpha=0.2)
        plt.ylabel('Pixels')
        plt.xlabel('Frame number')

        plt.figure(figsize=vip_figsize)
        b = int(np.sqrt(n_frames))
        la = 'Histogram'
        _ = plt.hist(shift_x, bins=b, alpha=0.5, label=la + ' shifts X')
        _ = plt.hist(shift_y, bins=b, alpha=0.5, label=la + ' shifts Y')
        plt.legend(loc='best')
        plt.ylabel('Bin counts')
        plt.xlabel('Pixels')

    if verbose:
        msg1 = 'MEAN X,Y: {:.3f}, {:.3f}'
        print(msg1.format(np.mean(shift_x), np.mean(shift_y)))
        msg2 = 'MEDIAN X,Y: {:.3f}, {:.3f}'
        print(msg2.format(np.median(shift_x), np.median(shift_y)))
        msg3 = 'STDDEV X,Y: {:.3f}, {:.3f}'
        print(msg3.format(np.std(shift_x), np.std(shift_y)))

    array_rec = np.array(array_rec) 
    return array_rec, shift_y, shift_x


def frame_center_radon(array, cropsize=101, hsize=0.4, step=0.01,
                       mask_center=None, nproc=None, satspots=False,
                       full_output=False, verbose=True, plot=True, debug=False):
    """ Finding the center of a broadband (co-added) frame with speckles and 
    satellite spots elongated towards the star (center). We use the radon
    transform implementation from scikit-image.
    
    Parameters
    ----------
    array : array_like
        Input 2d array or image.
    cropsize : odd int, optional
        Size in pixels of the cropped central area of the input array that will
        be used. It should be large enough to contain the satellite spots.
    hsize : float, optional
        Size of the box for the grid search. The frame is shifted to each 
        direction from the center in a hsize length with a given step.
    step : float, optional
        The step of the coordinates change.
    mask_center : None or int, optional
        If None the central area of the frame is kept. If int a centered zero 
        mask will be applied to the frame. By default the center isn't masked.
    nproc : int, optional
        Number of processes for parallel computing. If None the number of 
        processes will be set to cpu_count()/2. 
    verbose : bool optional
        Whether to print to stdout some messages and info.
    plot : bool, optional
        Whether to plot the radon cost function. 
    debug : bool, optional
        Whether to print and plot intermediate info.
    
    Returns
    -------
    optimy, optimx : float
        Values of the Y, X coordinates of the center of the frame based on the
        radon optimization.
    If full_output is True then the radon cost function surface is returned 
    along with the optimal x and y.
        
    Notes
    -----
    Based on Pueyo et al. 2014: http://arxiv.org/abs/1409.6388
    
    """
    from .cosmetics import frame_crop
    
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')

    if verbose:  start_time = time_ini()
    frame = array.copy()
    frame = frame_crop(frame, cropsize, verbose=False)
    listyx = np.linspace(start=-hsize, stop=hsize, num=2*hsize/step+1, 
                         endpoint=True)
    if not mask_center:
        radint = 0
    else:
        if not isinstance(mask_center, int):
            raise TypeError
        radint = mask_center
    
    coords = [(y,x) for y in listyx for x in listyx]
    cent, _ = frame_center(frame)
           
    frame = get_annulus(frame, radint, cent-radint)
                
    if debug:
        if satspots:
            samples = 10
            theta = np.hstack((np.linspace(start=40, stop=50, num=samples, 
                                           endpoint=False), 
                               np.linspace(start=130, stop=140, num=samples, 
                                           endpoint=False),
                               np.linspace(start=220, stop=230, num=samples, 
                                           endpoint=False),
                               np.linspace(start=310, stop=320, num=samples, 
                                           endpoint=False)))             
            sinogram = radon(frame, theta=theta, circle=True)
            pp_subplots(frame, sinogram)
            print(np.sum(np.abs(sinogram[cent,:])))
        else:
            theta = np.linspace(start=0, stop=360, num=cent*2, endpoint=False)
            sinogram = radon(frame, theta=theta, circle=True)
            pp_subplots(frame, sinogram)
            print(np.sum(np.abs(sinogram[cent,:])))

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores
    
    if satspots:
        costfkt = _radon_costf2
    else:
        costfkt = _radon_costf

    if nproc == 1:
        costf = []
        for coord in coords:
            res = costfkt(frame, cent, radint, coord)
            costf.append(res)
        costf = np.array(costf)
    elif nproc > 1:
        pool = Pool(processes=nproc)  
        res = pool.map(EFT, zip(itt.repeat(costfkt),
                                itt.repeat(frame), itt.repeat(cent),
                                itt.repeat(radint), coords))

        costf = np.array(res)
        pool.close()
        
    if verbose:  
        msg = 'Done {} radon transform calls distributed in {} processes'
        print(msg.format(len(coords), nproc))

    cost_bound = costf.reshape(listyx.shape[0], listyx.shape[0])
    if plot:
        plt.contour(cost_bound, cmap='CMRmap', origin='lower', lw=1, hold='on')
        plt.imshow(cost_bound, cmap='CMRmap', origin='lower', 
                   interpolation='nearest')
        plt.colorbar()
        plt.grid('off')
        plt.show()
        
    #argm = np.argmax(costf) # index of 1st max in 1d cost function 'surface' 
    #optimy, optimx = coords[argm]
    
    # maxima in the 2d cost function surface
    num_max = np.where(cost_bound == cost_bound.max())[0].shape[0]
    ind_maximay, ind_maximax = np.where(cost_bound == cost_bound.max())
    argmy = ind_maximay[int(np.ceil(num_max/2)) - 1]
    argmx = ind_maximax[int(np.ceil(num_max/2)) - 1]
    y_grid = np.array(coords)[:, 0].reshape(listyx.shape[0], listyx.shape[0])
    x_grid = np.array(coords)[:, 1].reshape(listyx.shape[0], listyx.shape[0])
    optimy = y_grid[argmy, 0] 
    optimx = x_grid[0, argmx]  
    
    if verbose: 
        print('Cost function max: {}'.format(costf.max()))
        print('Cost function # maxima: {}'.format(num_max))
        msg = 'Finished grid search radon optimization. Y={:.5f}, X={:.5f}'
        print(msg.format(optimy, optimx))
        timing(start_time)
    
    if full_output:
        return cost_bound, optimy, optimx
    else:
        return optimy, optimx


def _radon_costf(frame, cent, radint, coords):
    """ Radon cost function used in frame_center_radon().
    """
    frame_shifted = frame_shift(frame, coords[0], coords[1])
    frame_shifted_ann = get_annulus(frame_shifted, radint, cent-radint)
    theta = np.linspace(start=0, stop=360, num=frame_shifted_ann.shape[0],    
                    endpoint=False)
    sinogram = radon(frame_shifted_ann, theta=theta, circle=True)
    costf = np.sum(np.abs(sinogram[int(cent),:]))
    return costf
                
                
def _radon_costf2(frame, cent, radint, coords):
    """ Radon cost function used in frame_center_radon().
    """
    frame_shifted = frame_shift(frame, coords[0], coords[1])
    frame_shifted_ann = get_annulus(frame_shifted, radint, cent-radint)
    samples = 10
    theta = np.hstack((np.linspace(start=40, stop=50, num=samples,
                                   endpoint=False),
                       np.linspace(start=130, stop=140, num=samples,
                                   endpoint=False),
                       np.linspace(start=220, stop=230, num=samples,
                                   endpoint=False),
                       np.linspace(start=310, stop=320, num=samples,
                                   endpoint=False)))
    
    sinogram = radon(frame_shifted_ann, theta=theta, circle=True)
    costf = np.sum(np.abs(sinogram[int(cent),:]))
    return costf

      
def cube_recenter_radon(array, full_output=False, verbose=True, imlib='opencv',
                        interpolation='lanczos4', **kwargs):
    """ Recenters a cube looping through its frames and calling the 
    ``frame_center_radon`` function.
    
    Parameters
    ----------
    array : array_like
        Input 3d array or cube.
    full_output : {False, True}, bool optional
        If True the recentered cube is returned along with the y and x shifts.
    verbose : {True, False}, bool optional
        Whether to print timing and intermediate information to stdout.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    kwargs : dict
        Optional parameters (keywords and values) can be passed to the
        ``frame_center_radon`` function.
    
    Returns
    -------
    array_rec : array_like
        Recentered cube.
    If full_output is True:
    y, x : 1d array of floats
        Shifts in y and x.
     
    """
    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array')

    if verbose:
        start_time = time_ini()

    n_frames = array.shape[0]
    x = np.zeros((n_frames))
    y = np.zeros((n_frames))
    array_rec = array.copy()
    
    for i in Progressbar(range(n_frames), desc="frames", verbose=verbose):
        y[i], x[i] = frame_center_radon(array[i], verbose=False, plot=False, 
                                        **kwargs)
        array_rec[i] = frame_shift(array[i], y[i], x[i], imlib=imlib,
                                   interpolation=interpolation)
        
    if verbose:
        timing(start_time)

    if full_output:
        return array_rec, y, x
    else:
        return array_rec


def cube_recenter_dft_upsampling(array, cy_1=None, cx_1=None, negative=False,
                                 fwhm=4, subi_size=None, upsample_factor=100,
                                 imlib='opencv', interpolation='lanczos4',
                                 full_output=False, verbose=True,
                                 save_shifts=False, debug=False, plot=True):
    """ Recenters a cube of frames using the DFT upsampling method as 
    proposed in Guizar et al. 2008 and implemented in the
    ``register_translation`` function from scikit-image.
    
    The algorithm (DFT upsampling) obtains an initial estimate of the 
    cross-correlation peak by an FFT and then refines the shift estimation by 
    upsampling the DFT only in a small neighborhood of that estimate by means 
    of a matrix-multiply DFT.
    
    Parameters
    ----------
    array : array_like
        Input cube.
    cy_1, cx_1 : int, optional
        Coordinates of the center of the subimage for fitting a 2d Gaussian and
        centroiding the 1st frame. 
    negative : bool, optional
        If True the centroiding of the 1st frames is done with a negative 
        2d Gaussian fit.   
    fwhm : float, optional
        FWHM size in pixels.
    subi_size : int or None, optional
        Size of the square subimage sides in pixels, used to centroid to first
        frame. If subi_size is None then the first frame is assumed to be
        centered already.
    upsample_factor : int, optional
        Upsampling factor (default 100). Images will be registered to within
        1/upsample_factor of a pixel.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    full_output : bool, optional
        Whether to return 2 1d arrays of shifts along with the recentered cube 
        or not.
    verbose : bool, optional
        Whether to print to stdout the timing or not.
    save_shifts : bool, optional
        Whether to save the shifts to a file in disk.
    debug : bool, optional
        Whether to print to stdout the shifts or not.
    plot : bool, optional
        If True, the shifts are plotted.
    
    Returns
    -------
    array_recentered : array_like
        The recentered cube. Frames have now odd size.
    If full_output is True:
    y, x : array_like
        1d arrays with the shifts in y and x.     
    
    Notes
    -----
    Using the implementation from scikit-image of the algorithm described in
    Guizar-Sicairos et al. "Efficient subpixel image registration algorithms,"
    Opt. Lett. 33, 156-158 (2008). This algorithm registers two images (2-D
    rigid translation) within a fraction of a pixel specified by the user.
    Instead of computing a zero-padded FFT (fast Fourier transform), this code
    uses selective upsampling by a matrix-multiply DFT (discrete FT) to
    dramatically reduce computation time and memory without sacrificing
    accuracy. With this procedure all the image points are used to compute the
    upsampled cross-correlation in a very small neighborhood around its peak.
    
    """
    if verbose:
        start_time = time_ini()

    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array')

    n_frames, sizey, sizex = array.shape
    if subi_size is not None:
        if cx_1 is None or cy_1 is None:
            print('`cx_1` or `cy_1` not be provided')
            print('Using the coordinated of the 1st frame center for '
                  'the Gaussian 2d fit')
            cy_1, cx_1 = frame_center(array[0])
        if not isinstance(subi_size, int):
            raise ValueError('subi_size must be an integer or None')
        if subi_size < fwhm:
            raise ValueError('`subi_size` (value in pixels) is too small')
        if sizey % 2 == 0:
            if subi_size % 2 != 0:
                subi_size += 1
                print('`subi_size` is odd (while frame size is even)')
                print('Setting `subi_size` to {} pixels'.format(subi_size))
        else:
            if subi_size % 2 == 0:
                subi_size += 1
                print('`subi_size` is even (while frame size is odd)')
                print('Setting `subi_size` to {} pixels'.format(subi_size))
    
    n_frames = array.shape[0]
    x = np.zeros((n_frames))
    y = np.zeros((n_frames))
    array_rec = array.copy()

    cy, cx = frame_center(array[0])
    # Centroiding first frame with 2d gaussian and shifting
    msg0 = "The rest of the frames will be shifted by cross-correlation "
    msg0 += "wrt the 1st"
    if subi_size is not None:
        y1, x1 = _centroid_2dg_frame(array_rec, 0, subi_size, cy_1, cx_1,
                                     negative, debug, fwhm)
        x[0] = cx - x1
        y[0] = cy - y1
        array_rec[0] = frame_shift(array_rec[0], shift_y=y[0], shift_x=x[0],
                                   imlib=imlib, interpolation=interpolation)
        if verbose:
            msg = "Shift for first frame X,Y=({:.3f}, {:.3f})"
            print(msg.format(x[0], y[0]))
            print(msg0)
        if debug:
            titd = "original / shifted 1st frame subimage"
            pp_subplots(frame_crop(array[0], subi_size, verbose=False),
                        frame_crop(array_rec[0], subi_size, verbose=False),
                        grid=True, title=titd)
    else:
        if verbose:
            print("The first frame is assumed to be well centered.")
            print(msg0)
        x[0] = cx
        y[0] = cy
    
    # Finding the shifts with DTF upsampling of each frame wrt the first
    for i in Progressbar(range(1, n_frames), desc="frames", verbose=verbose):
        shift_yx, _, _ = register_translation(array_rec[0], array[i],
                                              upsample_factor=upsample_factor)
        y[i], x[i] = shift_yx
        array_rec[i] = frame_shift(array[i], shift_y=y[i], shift_x=x[i],
                                   imlib=imlib, interpolation=interpolation)

    if debug:
        print("\nShifts in X and Y")
        for i in range(n_frames):
            print(x[i], y[i])
        
    if verbose:
        timing(start_time)

    if plot:
        plt.figure(figsize=vip_figsize)
        plt.plot(y, 'o-', label='shifts in y', alpha=0.5)
        plt.plot(x, 'o-', label='shifts in x', alpha=0.5)
        plt.legend(loc='best')
        plt.grid('on', alpha=0.2)
        plt.ylabel('Pixels')
        plt.xlabel('Frame number')

        plt.figure(figsize=vip_figsize)
        b = int(np.sqrt(n_frames))
        la = 'Histogram'
        _ = plt.hist(x, bins=b, alpha=0.5, label=la + ' shifts X')
        _ = plt.hist(y, bins=b, alpha=0.5, label=la + ' shifts Y')
        plt.legend(loc='best')
        plt.ylabel('Bin counts')
        plt.xlabel('Pixels')

    if save_shifts: 
        np.savetxt('recent_dft_shifts.txt', np.transpose([y, x]), fmt='%f')
    if full_output:
        return array_rec, y, x
    else:
        return array_rec
  

def cube_recenter_2dfit(array, xy=None, fwhm=4, subi_size=5, model='gauss',
                        nproc=1, imlib='opencv', interpolation='lanczos4',
                        offset=None, negative=False, threshold=False,
                        save_shifts=False, full_output=False, verbose=True,
                        debug=False, plot=True):
    """ Recenters the frames of a cube. The shifts are found by fitting a 2d 
    Gaussian or Moffat to a subimage centered at ``xy``. This assumes the frames
    don't have too large shifts (>5px). The frames are shifted using the 
    function frame_shift().
    
    Parameters
    ----------
    array : array_like
        Input cube.
    xy : tuple of int
        Coordinates of the center of the subimage (wrt the original frame).
    fwhm : float or array_like
        FWHM size in pixels, either one value (float) that will be the same for
        the whole cube, or an array of floats with the same dimension as the 
        0th dim of array, containing the fwhm for each channel (e.g. in the case
        of an ifs cube, where the fwhm varies with wavelength)
    subi_size : int, optional
        Size of the square subimage sides in pixels.
    model : str, optional
        Sets the type of fit to be used. 'gauss' for a 2d Gaussian fit and
        'moff' for a 2d Moffat fit.
    nproc : int or None, optional
        Number of processes (>1) for parallel computing. If 1 then it runs in 
        serial. If None the number of processes will be set to (cpu_count()/2).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_shift`` function.
    offset : tuple of floats, optional
        If None the region of the frames used for the 2d Gaussian/Moffat fit is
        shifted to the center of the images (2d arrays). If a tuple is given it
        serves as the offset of the fitted area wrt the center of the 2d arrays.
    negative : bool, optional
        If True a negative 2d Gaussian/Moffat fit is performed.
    threshold : bool, optional
        If True the background pixels (estimated using sigma clipped statistics)
        will be replaced by small random Gaussian noise.
    save_shifts : bool, optional
        Whether to save the shifts to a file in disk.
    full_output : bool, optional
        Whether to return 2 1d arrays of shifts along with the recentered cube 
        or not.
    verbose : bool, optional
        Whether to print to stdout the timing or not.
    debug : bool, optional
        If True the details of the fitting are shown. Won't work when the cube
        contains >20 frames (as it might produce an extremely long output).
    plot : bool, optional
        If True, the shifts are plotted.
        
    Returns
    -------
    array_recentered : array_like
        The recentered cube. Frames have now odd size.
    If full_output is True:
    y, x : array_like
        1d arrays with the shifts in y and x. 
    
    """
    if verbose:
        start_time = time_ini()

    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array')

    n_frames, sizey, sizex = array.shape

    if not isinstance(subi_size, int):
        raise ValueError('`subi_size` must be an integer')
    if subi_size < fwhm:
        raise ValueError('`subi_size` (value in pixels) is too small')
    if sizey % 2 == 0:
        if subi_size % 2 != 0:
            subi_size += 1
            print('`subi_size` is odd (while frame size is even)')
            print('Setting `subi_size` to {} pixels'.format(subi_size))
    else:
        if subi_size % 2 == 0:
            subi_size += 1
            print('`subi_size` is even (while frame size is odd)')
            print('Setting `subi_size` to {} pixels'.format(subi_size))

    if isinstance(fwhm, (float, int)):
        fwhm = np.ones(n_frames) * fwhm

    if debug and array.shape[0] > 20:
        msg = 'Debug with a big array will produce a very long output. '
        msg += 'Try with less than 20 frames in debug mode'
        raise RuntimeWarning(msg)

    if xy is not None:
        pos_x, pos_y = xy
        if not isinstance(pos_x, int) or not isinstance(pos_y, int):
            raise TypeError('`xy` must be a tuple of integers')
    else:
        pos_y, pos_x = frame_center(array[0])

    cy, cx = frame_center(array[0])
    array_recentered = np.empty_like(array)

    if model == 'gauss':
        func = _centroid_2dg_frame
    elif model == 'moff':
        func = _centroid_2dm_frame
    else:
        raise ValueError('model not recognized')
    
    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    if nproc == 1:
        res = []
        print('2d Gauss-fitting')
        for i in Progressbar(range(n_frames), desc="frames", verbose=verbose):
            res.append(func(array, i, subi_size, pos_y, pos_x, negative, debug,
                            fwhm[i], threshold))
        res = np.array(res)
    elif nproc > 1:
        pool = Pool(processes=nproc)  
        res = pool.map(EFT, zip(itt.repeat(func), itt.repeat(array),
                                range(n_frames), itt.repeat(subi_size),
                                itt.repeat(pos_y), itt.repeat(pos_x),
                                itt.repeat(negative), itt.repeat(debug), fwhm,
                                itt.repeat(threshold)))
        res = np.array(res)
        pool.close()
    y = cy - res[:, 0]
    x = cx - res[:, 1]

    if offset is not None:
        offx, offy = offset
        y -= offy
        x -= offx
    
    for i in Progressbar(range(n_frames), desc="shifting", verbose=verbose):
        if debug:
            print("\nShifts in X and Y")
            print(x[i], y[i])
        array_recentered[i] = frame_shift(array[i], y[i], x[i], imlib=imlib,
                                          interpolation=interpolation)
        
    if verbose:
        timing(start_time)

    if plot:
        plt.figure(figsize=vip_figsize)
        plt.plot(y, 'o-', label='shifts in y', alpha=0.5)
        plt.plot(x, 'o-', label='shifts in x', alpha=0.5)
        plt.legend(loc='best')
        plt.grid('on', alpha=0.2)
        plt.ylabel('Pixels')
        plt.xlabel('Frame number')

        plt.figure(figsize=vip_figsize)
        b = int(np.sqrt(n_frames))
        la = 'Histogram'
        _ = plt.hist(x, bins=b, alpha=0.5, label=la + ' shifts X')
        _ = plt.hist(y, bins=b, alpha=0.5, label=la + ' shifts Y')
        plt.legend(loc='best')
        plt.ylabel('Bin counts')
        plt.xlabel('Pixels')

    if save_shifts: 
        np.savetxt('recent_gauss_shifts.txt', np.transpose([y, x]), fmt='%f')
    if full_output:
        return array_recentered, y, x
    else:
        return array_recentered


def _centroid_2dg_frame(cube, frnum, size, pos_y, pos_x, negative, debug,
                        fwhm, threshold=False):
    """ Finds the centroid by using a 2d gaussian fitting in one frame from a
    cube.
    """
    sub_image, y1, x1 = get_square(cube[frnum], size=size, y=pos_y, x=pos_x,
                                   position=True)
    # negative gaussian fit
    if negative:
        sub_image = -sub_image + np.abs(np.min(-sub_image))

    y_i, x_i = fit_2dgaussian(sub_image, crop=False, fwhmx=fwhm, fwhmy=fwhm,
                              threshold=threshold, sigfactor=1, debug=debug)
    y_i = y1 + y_i
    x_i = x1 + x_i
    return y_i, x_i


def _centroid_2dm_frame(cube, frnum, size, pos_y, pos_x, negative, debug,
                        fwhm, threshold=False):
    """ Finds the centroid by using a 2d moffat fitting in one frame from a
    cube.
    """
    sub_image, y1, x1 = get_square(cube[frnum], size=size, y=pos_y, x=pos_x,
                                   position=True)
    # negative fit
    if negative:
        sub_image = -sub_image + np.abs(np.min(-sub_image))

    y_i, x_i = fit_2dmoffat(sub_image, crop=False, fwhm=fwhm, debug=debug,
                            threshold=threshold, full_output=False)
    y_i = y1 + y_i
    x_i = x1 + x_i
    return y_i, x_i


# TODO: make parameter names match the API
def cube_recenter_via_speckles(cube_sci, cube_ref=None, alignment_iter=5,
                               gammaval=1, min_spat_freq=0.5, max_spat_freq=3,
                               fwhm=4, debug=False, negative=True,
                               recenter_median=False, subframesize=20,
                               imlib='opencv', interpolation='bilinear',
                               save_shifts=False, plot=True):
    """ Registers frames based on the median speckle pattern. Optionally centers 
    based on the position of the vortex null in the median frame. Images are 
    filtered to isolate speckle spatial frequencies.

    Parameters
    ----------
    cube_sci : array_like  
        Science cube.
    cube_ref : array_like
        Reference cube (e.g. for NIRC2 data in RDI mode). 
    alignment_iter : int, optional  
        Number of alignment iterations (recomputes median after each iteration). 
    gammaval : int, optional 
        Applies a gamma correction to emphasize speckles (useful for faint 
        stars).
    min_spat_freq : float, optional 
        Spatial frequency for high pass filter. 
    max_spat_freq : float, optional 
        Spatial frequency for low pass filter. 
    fwhm : float, optional 
        Full width at half maximum. 
    debug : bool, optional
        Outputs extra info.
    negative : bool, optional
        Use a negative gaussian fit to determine the center of the median frame.
    recenter_median : bool, optional
        Recenter the frames at each iteration based on the gaussian fit.
    subframesize : int, optional
        Sub-frame window size used. Should cover the region where speckles are 
        the dominant noise source. 
    imlib : str, optional 
        Image processing library to use. 
    interpolation : str, optional 
        Interpolation method to use.
    save_shifts : bool, optional
        Whether to save the shifts to a file in disk.
    plot : bool, optional
        If True, the shifts are plotted.

    Returns
    -------
    array_shifted : array_like
        Shifted 2d array.

    If cube_ref is not None, returns:
    cube_reg_sci: Registered science cube.
    cube_reg_ref: Ref. cube registered to science frames.
    cum_x_shifts_sci: Vector of x shifts for science frames.
    cum_y_shifts_sci: Vector of y shifts for science frames.
    cum_x_shifts_ref: Vector of x shifts for ref. frames.
    cum_y_shifts_ref: Vector of y shifts for ref. frames.

    Otherwise, returns: cube_reg_sci, cum_x_shifts_sci, cum_y_shifts_sci
    """
    n, y, x = cube_sci.shape

    if not subframesize < y/2.:
        raise ValueError('`Subframesize` is too large')

    if cube_ref is not None:
        ref_star = True
        nref = cube_ref.shape[0]
    else:
        ref_star = False

    cube_sci_subframe = cube_crop_frames(cube_sci, subframesize, verbose=False)
    if ref_star:
        cube_ref_subframe = cube_crop_frames(cube_ref, subframesize,
                                             verbose=False)

    ceny, cenx = frame_center(cube_sci_subframe[0])
    print('Sub frame shape: {}'.format(cube_sci_subframe.shape))
    print('Center pixel: ({}, {})'.format(ceny, cenx))

    # Filtering cubes. Will be used for alignment purposes
    cube_sci_lpf = cube_sci_subframe.copy()
    if ref_star:
        cube_ref_lpf = cube_ref_subframe.copy()

    cube_sci_lpf = cube_sci_lpf + np.abs(np.min(cube_sci_lpf))
    if ref_star:
        cube_ref_lpf = cube_ref_lpf + np.abs(np.min(cube_ref_lpf))

    median_size = int(fwhm * max_spat_freq)
    # Remove spatial frequencies <0.5 lam/D and >3lam/D to isolate speckles
    cube_sci_hpf = cube_filter_highpass(cube_sci_lpf, 'median-subt',
                                        median_size=median_size, verbose=False)
    cube_sci_lpf = cube_filter_lowpass(cube_sci_hpf, 'gauss',
                                       fwhm_size=median_size, verbose=False)

    if ref_star:
        cube_ref_hpf = cube_filter_highpass(cube_ref_lpf, 'median-subt',
                                            median_size=median_size,
                                            verbose=False)
        cube_ref_lpf = cube_filter_lowpass(cube_ref_hpf, 'gauss',
                                           fwhm_size=min_spat_freq * fwhm,
                                           verbose=False)

    if ref_star:
        alignment_cube = np.zeros((1 + n + nref, subframesize, subframesize))
        alignment_cube[1:(n + 1), :, :] = cube_sci_lpf
        alignment_cube[(n + 1):(n + 2 + nref), :, :] = cube_ref_lpf
    else:
        alignment_cube = np.zeros((1 + n, subframesize, subframesize))
        alignment_cube[1:(n + 1), :, :] = cube_sci_lpf

    n_frames = alignment_cube.shape[0]  # 1+n or 1+n+nref
    cum_y_shifts = 0
    cum_x_shifts = 0

    for i in range(alignment_iter):
        alignment_cube[0] = np.median(alignment_cube[1:(n + 1)], axis=0)
        if recenter_median:
            # Recenter the median frame using a neg. gaussian fit
            sub_image, y1, x1 = get_square(alignment_cube[0], size=int(fwhm),
                                           y=ceny, x=cenx, position=True)
            if negative:
                sub_image = -sub_image + np.abs(np.min(-sub_image))
            y_i, x_i = fit_2dgaussian(sub_image, crop=False, threshold=False,
                                      sigfactor=1, debug=debug)
            yshift = ceny - (y1 + y_i)
            xshift = cenx - (x1 + x_i)

            alignment_cube[0] = frame_shift(alignment_cube[0, :, :], yshift,
                                            xshift, imlib=imlib,
                                            interpolation=interpolation)

        # center the cube with stretched values
        cube_stret = np.log10((np.abs(alignment_cube) + 1) ** gammaval)
        _, y_shift, x_shift = cube_recenter_dft_upsampling(cube_stret, ceny,
                                            cenx, fwhm=fwhm, subi_size=None,
                                            full_output=True, verbose=False,
                                            plot=False)
        sqsum_shifts = np.sum(np.sqrt(y_shift ** 2 + x_shift ** 2))
        print('Square sum of shift vecs: ' + str(sqsum_shifts))

        for j in range(1, n_frames):
            alignment_cube[j] = frame_shift(alignment_cube[j], y_shift[j],
                                            x_shift[j], imlib=imlib,
                                            interpolation=interpolation)

        cum_y_shifts += y_shift
        cum_x_shifts += x_shift

    cube_reg_sci = cube_sci.copy()
    cum_y_shifts_sci = cum_y_shifts[1:(n + 1)]
    cum_x_shifts_sci = cum_x_shifts[1:(n + 1)]
    for i in range(n):
        cube_reg_sci[i] = frame_shift(cube_sci[i], cum_y_shifts_sci[i],
                                      cum_x_shifts_sci[i], imlib=imlib,
                                      interpolation=interpolation)

    if plot:
        plt.figure(figsize=vip_figsize)
        plt.plot(cum_x_shifts_sci, 'o-', label='Shifts in x', alpha=0.5)
        plt.plot(cum_y_shifts_sci, 'o-', label='Shifts in y', alpha=0.5)
        plt.legend(loc='best')
        plt.grid('on', alpha=0.2)
        plt.ylabel('Pixels')
        plt.xlabel('Frame number')

        plt.figure(figsize=vip_figsize)
        b = int(np.sqrt(n))
        la = 'Histogram'
        _ = plt.hist(cum_x_shifts_sci, bins=b, alpha=0.5, label=la+' shifts X')
        _ = plt.hist(cum_y_shifts_sci, bins=b, alpha=0.5, label=la+' shifts Y')
        plt.legend(loc='best')
        plt.ylabel('Bin counts')
        plt.xlabel('Pixels')

    if ref_star:
        cube_reg_ref = cube_ref.copy()
        cum_y_shifts_ref = cum_y_shifts[(n + 1):]
        cum_x_shifts_ref = cum_x_shifts[(n + 1):]
        for i in range(nref):
            cube_reg_ref[i] = frame_shift(cube_ref[i], cum_y_shifts_ref[i],
                                          cum_x_shifts_ref[i], imlib=imlib,
                                          interpolation=interpolation)

    if save_shifts:
        np.savetxt('recent_gauss_shifts.txt',
                   np.transpose([cum_y_shifts_sci, cum_x_shifts_sci]), fmt='%f')
    if ref_star:
        return (cube_reg_sci, cube_reg_ref, cum_x_shifts_sci, cum_y_shifts_sci,
                cum_x_shifts_ref, cum_y_shifts_ref)
    else:
        return cube_reg_sci, cum_x_shifts_sci, cum_y_shifts_sci



