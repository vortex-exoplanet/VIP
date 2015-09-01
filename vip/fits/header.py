"""
Created in February 2015, last updated in August 2015

Module of subroutines used to manipulate ESO fits headers.
"""

__author__ = 'V. Christiaens @ U.Chile/ULg'

import numpy as np


def coordinates(header,j):
    """
    FUNCTION TO READ HEADER AND RETURNS THE COORDINATE OF EACH POINT OF AXIS j

    Parameter:
    ----------
    header: ESO-like header

    Returns:
    --------
    val: 1d array
        Contains the value of each coordinate along the j-th axis.

    Typically:
    if j=1, returns RA coordinates of each column of px along x (only true for 
    derotated frames/cubes; i.e. with North up and East left);
    if j=2, returns DEC coordinates of each row of px along y (only true for 
    derotated frames/cubes; i.e. with North up and East left) ; 
    if j=3, returns the wavelength range

    """
    N=header['NAXIS'+str(j)];
    val=np.zeros(N);
    for i in range(0,N):
        val[i]=(i+1-float(header['CRPIX'+str(j)])) * \
        float(header['CD'+str(j)+'_'+str(j)]) + float(header['CRVAL'+str(j)])
    return val


def coordinate(header,j,i):
    """
    FUNCTION TO READ HEADER AND FIND THE COORDINATE OF POINT i OF AXIS j

    Parameter:
    ----------
    header: ESO-like header

    Returns:
    --------
    val: float

    Examples:
    ---------
    if j=1, returns coordinate in RA of column x = i (only true for 
    derotated frames/cubes; i.e. with North up and East left);
    if j=2, returns coordinate in DEC of column y = i (only true for 
    derotated frames/cubes; i.e. with North up and East left); 
    if j=3, returns coordinate in microns of channel z = i
    """

    val = (i+1-float(header['CRPIX'+str(j)])) * \
          float(header['CD'+str(j)+'_'+str(j)]) + float(header['CRVAL'+str(j)])
    return val


def genlbda(header):
    """
    FUNCTION TO GENERATE A VECTOR WITH ALL THE WAVELENGTH VALUES

    Parameter:
    ----------
    header: ESO-like header

    Returns:
    --------
    lbda: 1d array
        Contains the wavelength (in same unit as header - usually um) for each 
        channel of the header's cube
    """

    lbda= coordinates(header,3)

    return lbda


def genfwhm(header):
    """
    FUNCTION TO GENERATE A VECTOR WITH ALL THE FWHM (in pixels

    Parameter:
    ----------
    header: ESO-like header

    Returns:
    --------
    fwhm: 1d array
        Contains the fwhm (in pixels) for each channel of the header's cube
    """

    if 'um' in header['CUNIT3']:
        lbda = 1e-6*genlbda(header)
    else:
        raise ValueError('The unit of cdelt 3 in header is not recognized')

    if 'VLT' in header['TELESCOP']:
        diam = 8.2
    else:
        raise ValueError('ESO telescope not recognized. Add "elif" case.')

    cdelt = abs(header['CDELT1'])
    if 'deg' in header['CUNIT1']:
        plsc = cdelt*3600
    else:
        raise ValueError('The unit of cdelt 1 in header is not recognized')

    fwhm = (lbda/diam)*206265/plsc

    return fwhm


def hcubetoim(header,dim4=False):
    """
    FUNCTION TO CONVERT A HEADER FROM 3D (or 4D) to 2D (useful for output 
    images like median,mean, etc.).
    
    Parameter:
    ----------
    header: ESO-like header
    dim4: bool, {False,True} optional
        Whether the input header is associated to a dimension 4 cube.

    Returns:
    --------
    hplane: header
        The modified header.
    """

    hplane=header
    #cards=hplane.ascardlist()
    hplane['NAXIS']=2
    del hplane['NAXIS3']
    if dim4: del hplane['NAXIS4']
    del hplane['CRPIX3']
    del hplane['CRVAL3']
    del hplane['CD1_3']
    del hplane['CD2_3']
    del hplane['CD3_3']
    del hplane['CD3_1']
    del hplane['CD3_2']
    del hplane['CUNIT3']
    try: del hplane['CDELT3']
    except: pass
    del hplane['CTYPE3']
    del hplane['EXTEND']
    del hplane['BITPIX']
    del hplane['CDBFILE'] 
    del hplane['UTC']
    del hplane['INSTRUME']
    del hplane['OBJECT']
    del hplane['COMMENT']
    del hplane['OBSERVER']
    del hplane['DATASUM']
    del hplane['DATAMD5']
    del hplane['ARCFILE']
    del hplane['SIMPLE']
    del hplane['PIPEFILE']
    return hplane


def himtocube(header,nz_fin,crpix3,crval3,cunit3='um',cdelt3=0.005,
              ctype3='WAVE'):
    """
    FUNCTION TO CONVERT A HEADER FROM 2D to 3D (useful for output cubes made by
    concatenation of postprocessing images).
    
    !!! Important: some mandatory keys (e.g. NAXIS3) have to be located at the 
    right place in the header. hdu.verify('fix') handles that. You have to run 3
    additional lines after you call himtocube: 
        hplane = himtocube(header,nz_fin,crpix3,crval3)
        hdu_new = fits.PrimaryHDU(cube, hplane)
        hdu_new.verify('fix')
        hplane = hdu_new.header

    Parameters:
    -----------
    header: ESO-like header
    nz_fin: int
        number of frames along the 3rd axis
    crpix3: int
        reference pixel along the 3rd axis
    crval3: float
        reference pixel value along the 3rd axis
    cunit3: string, optional
        unit of the 3rd axis (e.g. 'um')
    cdelt3: float, optional
        increment along the 3rd axis
    ctype3: string, optional
        type of the 3rd axis (e.g. 'WAVE')

    Returns:
    --------
    hplane: header
        The modified header
    """
    hplane=header
    #cards=hplane.ascardlist()
    hplane['NAXIS']=3
    hplane['NAXIS3']=nz_fin
    hplane['CRPIX3']=crpix3
    hplane['CRVAL3']=crval3
    hplane['CD1_3'] = 0.
    hplane['CD2_3'] = 0.
    hplane['CD3_3'] = cdelt3
    hplane['CD3_1'] = 0.
    hplane['CD3_2'] = 0.
    hplane['CUNIT3'] = cunit3
    hplane['CDELT3'] = cdelt3
    hplane['CTYPE3'] = ctype3

    return hplane


def hsmallercube(header,n_subtr_chan_init,n_subtr_chan_fin):
    """
    FUNCTION TO CHANGE A HEADER WHEN THE WAVELENGTH RANGE IS REDUCED
    Note: this function is different than hdifferentZ in the sense that it 
    assumes that the cube remains a spectral cube along the z-axis.     

    Parameters:
    -----------
    header: ESO-like header
    n_subtr_chan_init: int
       index of the first channel that is kept
    n_subtr_chan_fin: int
       index of the last channel that is kept

    Returns:
    --------
    hplane: header
       The modified header

    """

    hplane=header.copy()
    n_z = hplane['NAXIS3']
    if (n_subtr_chan_init >= hplane['CRPIX3'] 
        or n_z-n_subtr_chan_fin < hplane['CRPIX3']):
        lbda = genlbda(header)
        hplane['CRVAL3']=lbda[n_subtr_chan_init]
        hplane['CRPIX3']=1
    else:
        hplane['CRPIX3']=hplane['CRPIX3']-n_subtr_chan_init
    nz_fin = n_z-n_subtr_chan_init-n_subtr_chan_fin
    hplane['NAXIS3']= nz_fin
    return hplane


def hdifferentZ(header, nz_fin, crpix3=1, crval3=1, cunit3='#', cdelt3=1,
                ctype3='cube index'):
    """
    FUNCTION TO CHANGE A HEADER WHEN THE Z DIMENSION IS CHANGED (e.g. from 
    wavelength to cube index)

    Parameters:
    -----------
    header: ESO-like header
    nz_fin: int
        number of frames along the 3rd axis
    crpix3: int, optional
        reference pixel along the 3rd axis
    crval3: float, optional
        reference pixel value along the 3rd axis
    cunit3: string, optional
        unit of the 3rd axis
    cdelt3: float, optional
        increment along the 3rd axis
    ctype3: string, optional
        type of the 3rd axis

    Returns:
    --------
    hplane: header
        The modified header
    
    """
    hplane=header
    hplane['NAXIS3'] = nz_fin
    hplane['CRPIX3'] = crpix3
    hplane['CRVAL3'] = crval3
    hplane['CUNIT3'] = cunit3
    hplane['CDELT3'] = cdelt3
    hplane['CD1_3'] = 0.
    hplane['CD2_3'] = 0.
    hplane['CD3_3'] = cdelt3
    hplane['CD3_1'] = 0.
    hplane['CD3_2'] = 0.
    hplane['CTYPE3'] = ctype3

    return hplane


def hdifferentXorY(header,ny_fin,nx_fin):
    """
    FUNCTION TO CHANGE A HEADER WHEN THE Y OR X DIMENSIONS ARE CHANGED.
    IMPORTANT: IT ASSUMES THE STAR HAS BEEN PLACED ON THE CENTRAL PIXEL (via 
    frame_center.py), and the frame has been derotated to match North up.

    Parameters:
    -----------
    header: ESO-like header
    ny_fin: int
       final dimension along y
    nx_fin: int
       final dimension along x

    Returns:
    --------
    hplane: header
       The modified header
    """
    hplane=header.copy()
    hplane['NAXIS2']= ny_fin
    hplane['NAXIS1']= nx_fin
    hplane['CRPIX2']= int(ny_fin/2.)   #the header is 1-based (so no need of 
                                       #subtracting 1 to n/2)
    hplane['CRPIX1']= int(nx_fin/2.)
    hplane['CRVAL2']= header['DEC']
    hplane['CRVAL1']= header['RA']
    return hplane

