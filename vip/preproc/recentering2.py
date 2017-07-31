#! /usr/bin/env python

__author__ = 'G. Ruane'
__all__ = ['register_and_center_via_speckles']

import numpy as np
from . import cube_recenter_dft_upsampling, frame_shift, cube_crop_frames
from ..var import frame_filter_gaussian2d,frame_filter_lowpass,get_square_robust,fit_2dgaussian,frame_center


def register_and_center_via_speckles(cube_sci, cube_ref=None, AlignmentIterations = 5, gammaval = 1,
min_spat_freq = 0.5, max_spat_freq = 3, fwhm = 8., debug = False , NegFit = True, recenter_median = True, subframesize = 151, imlib='opencv',interpolation='bilinear'):
    """ Registers frames based on the median speckle pattern. Optionally centers based on the position of the vortex null in the median frame. Images are filtered to isolate speckle spatial frequencies.
        
        Parameters
        ----------
        cube_sci: Science cube.
        cube_ref: Reference cube. 
        AlignmentIterations: Number of alignment iterations (recomputes median after each iteration) 
        gammaval: Applies a gamma correction to emphasize speckles (useful for faint stars)
        min_spat_freq: Spatial frequency for high pass filter. 
        max_spat_freq: Spatial frequency for low pass filter. 
        fwhm: Full width at half maximum 
        debug: Outputs extra info
        NegFit: Use a negative gaussian fit to determine the center of the median frame (True of False).
        recenter_median: Recenter the frames at each iteration based on the gaussian fit (True of False).
        subframesize: Sub-frame window size used. Should cover the region where speckles are dominant noise source. 
        imlib: Image processing library to use. 
        interpolation: Interpolation method to use.

        
        Returns
        -------
        array_shifted : array_like
        Shifted 2d array.
        
        if cube_ref is not None, returns
        cube_reg_sci: Registered science cube.
        cube_reg_ref: Ref. cube registered to science frames.
        cum_x_shifts_sci: Vector of x shifts for science frames.
        cum_y_shifts_sci: Vector of y shifts for science frames.
        cum_x_shifts_ref: Vector of x shifts for ref. frames.
        cum_y_shifts_ref: Vector of y shifts for ref. frames.
        
        else, returns
        cube_reg_sci
        cum_x_shifts_sci
        cum_y_shifts_sci
        
        Notes
        -----
        Regarding the imlib parameter: 'ndimage-fourier', does a fourier shift
        operation and preserves better the pixel values (therefore the flux and
        photometry). 'ndimage-fourier' is used by default from VIP version 0.5.3.
        Interpolation based shift ('opencv' and 'ndimage-interp') is faster than the
        fourier shift. 'opencv' could be used when speed is critical and the flux
        preservation is not that important.
        
        """
    if cube_ref is not None:
        refStar = True
    else:
        refStar = False

    cube_sci_subframe=cube_crop_frames(cube_sci, subframesize, verbose=False) 
    if(refStar):
        cube_ref_subframe=cube_crop_frames(cube_ref, subframesize, verbose=False) 
    
    ceny,cenx = frame_center(cube_sci_subframe[0,:,:])
    print 'sub frame is '+str(cube_sci_subframe.shape[1])+'x'+str(cube_sci_subframe.shape[2])
    print 'center pixel is ('+str(ceny)+', '+str(cenx)+')'
    
    # Make a copy of the sci and ref frames, filter them, will be used for alignment purposes
    cube_sci_lpf = cube_sci_subframe.copy()
    if(refStar):
        cube_ref_lpf = cube_ref_subframe.copy()
    
    cube_sci_lpf = cube_sci_lpf - np.min(cube_sci_lpf)
    if(refStar):
        cube_ref_lpf = cube_ref_lpf - np.min(cube_ref_lpf)
    
    # Remove spatial frequencies <0.5 lam/D and >3lam/D to isolate speckles
    for i in range(cube_sci.shape[0]):
        cube_sci_lpf[i,:,:] = cube_sci_lpf[i,:,:] - frame_filter_lowpass(cube_sci_lpf[i,:,:], 'median', median_size=fwhm*max_spat_freq)
    if(refStar):
        for i in range(cube_ref.shape[0]):
            cube_ref_lpf[i,:,:] = cube_ref_lpf[i,:,:] - frame_filter_lowpass(cube_ref_lpf[i,:,:], 'median', median_size=fwhm*max_spat_freq)
    
    for i in range(cube_sci.shape[0]):
        cube_sci_lpf[i,:,:] = frame_filter_gaussian2d(cube_sci_lpf[i,:,:], min_spat_freq*fwhm)
    if(refStar):
        for i in range(cube_ref.shape[0]):
            cube_ref_lpf[i,:,:] = frame_filter_gaussian2d(cube_ref_lpf[i,:,:], min_spat_freq*fwhm) 
    
    if(refStar):
        alignment_cube = np.zeros((1+cube_sci.shape[0]+cube_ref.shape[0],cube_sci_subframe.shape[1],cube_sci_subframe.shape[2]))
        alignment_cube[1:(cube_sci.shape[0]+1),:,:]=cube_sci_lpf
        alignment_cube[(cube_sci.shape[0]+1):(cube_sci.shape[0]+2+cube_ref.shape[0]),:,:]=cube_ref_lpf
    else:
        alignment_cube = np.zeros((1+cube_sci.shape[0],cube_sci_subframe.shape[1],cube_sci_subframe.shape[2]))
        alignment_cube[1:(cube_sci.shape[0]+1),:,:]=cube_sci_lpf
    
    n_frames = alignment_cube.shape[0] # number of sci+ref frames + 1 for the median
    
    cum_y_shifts = 0
    cum_x_shifts = 0
    
    for i in range(AlignmentIterations):
        
        alignment_cube[0,:,:]=np.median(alignment_cube[1:(cube_sci.shape[0]+1),:,:],axis=0) 
        if(recenter_median):
            ## Recenter the median frame using a neg. gaussian fit
            sub_image, y1, x1 = get_square_robust(alignment_cube[0,:,:], size=int(fwhm)+1, y=ceny,x=cenx, position=True)
            if(NegFit):
                sub_image = -sub_image + np.abs(np.min(-sub_image))       
            y_i, x_i = fit_2dgaussian(sub_image, crop=False, threshold=False,sigfactor=1, debug=debug)
            yshift = ceny - (y1 + y_i)
            xshift = cenx - (x1 + x_i)
    
            print yshift,xshift
            alignment_cube[0,:,:] = frame_shift(alignment_cube[0,:,:], yshift, xshift, imlib=imlib, interpolation=interpolation)
        
        # center the cube with stretched values
        _,y_shift,x_shift = cube_recenter_dft_upsampling(np.log10((abs(alignment_cube)+1)**(gammaval)), ceny, cenx, fwhm=fwhm, 
                                         subi_size=None, full_output=True, verbose=False, save_shifts=False, debug=False)   
        
        print '\nSquare sum of shift vecs: '+str(np.sum(np.sqrt(y_shift**2+x_shift**2)))
        
        for i in xrange(1, n_frames):
            alignment_cube[i] = frame_shift(alignment_cube[i], y_shift[i], x_shift[i], imlib=imlib, interpolation=interpolation)
        
        cum_y_shifts = cum_y_shifts + y_shift
        cum_x_shifts = cum_x_shifts + x_shift
    
    cum_y_shifts_sci = cum_y_shifts[1:(cube_sci.shape[0]+1)]
    cum_x_shifts_sci = cum_x_shifts[1:(cube_sci.shape[0]+1)]
    
    cube_reg_sci = cube_sci.copy()
    for i in range(cube_sci.shape[0]):
        cube_reg_sci[i] = frame_shift(cube_sci[i], cum_y_shifts_sci[i], cum_x_shifts_sci[i], imlib=imlib, interpolation=interpolation)
        
    if(refStar):
        cube_reg_ref = cube_ref.copy()
        cum_y_shifts_ref = cum_y_shifts[(cube_sci.shape[0]+1):]
        cum_x_shifts_ref = cum_x_shifts[(cube_sci.shape[0]+1):]
        for i in range(cube_ref.shape[0]):
            cube_reg_ref[i] = frame_shift(cube_ref[i], cum_y_shifts_ref[i], cum_x_shifts_ref[i], imlib=imlib, interpolation=interpolation)

    if cube_ref is not None:
        return cube_reg_sci,cube_reg_ref, cum_x_shifts_sci, cum_y_shifts_sci, cum_x_shifts_ref, cum_y_shifts_ref
    else:
        return cube_reg_sci, cum_x_shifts_sci, cum_y_shifts_sci

