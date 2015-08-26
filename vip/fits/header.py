"""
Created in February 2015, last updated in August 2015

Module of subroutines used to manipulate fits headers.
"""

__author__ = 'V. Christiaens @ UChile/ULg'

import numpy as np


def coordinates(header,j):
    """
    FUNCTION TO READ HEADER h AND FIND THE COORDINATE OF EACH POINT OF AXIS j
    Example:
    if j=1, [val]=RA; if j=2, [val]=DEC; if j=3, [val]=microns
    """
    N=header['NAXIS'+str(j)];
    val=np.zeros(N);
    for i in range(0,N):
        val[i] = (i+1-float(header['CRPIX'+str(j)]))*float(header['CD'+str(j)+'_'+str(j)]) + float(header['CRVAL'+str(j)]);
    return val


def coordinate(header,j,i):
    """
    FUNCTION TO READ HEADER h AND FIND THE COORDINATE OF POINT i OF AXIS j
    Example:
    if j=1, [val]=RA; if j=2, [val]=DEC; if j=3, [val]=microns
    """
    val = (i+1-float(header['CRPIX'+str(j)]))*float(header['CD'+str(j)+'_'+str(j)]) + float(header['CRVAL'+str(j)]);
    return val


def genlbda(header):
    """
    FUNCTION TO GENERATE A VECTOR WITH ALL THE WAVELENGTH VALUES - author: Vachail Salinas

    Parameter:
    ----------
    header:header_like
    """
    lbda= np.array([header['CD3_3']*(x - (header['CRPIX3']-1)) + header['CRVAL3'] for x in range(header['NAXIS3'])])   # in microns (units of header)  
    return lbda


def hcubetoim(header,dim4=False):
    """
    FUNCTION TO CONVERT A HEADER FROM 3D to 2D (useful for output images like median,mean, etc.).
    
    Parameter:
    ----------
    header:header_like
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
    #h=0
    #a= len(cards)
    '''
    for x in range(a):
    s= str(cards[x-h])
    s1=s[:8]
    if (s1 == 'HIERARCH'):
    del hplane[x-h]
    h+=1
    '''
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


def himtocube(header,nz_fin,crpix3,crval3,cunit3='um',cdelt3=0.005,ctype3='WAVE'):
    """
    FUNCTION TO CONVERT A HEADER FROM 2D to 3D (useful for output cubes made by concatenation of postprocessing images).
    
    !!! Important, you have to run the three following lines after you call this function. Indeed, some mandatory keys (e.g. NAXIS3) have to be located at the right place in the header. hdu.verify('fix') handles that. Let's say the header you just modified is called hplane, you overwrite it after correction: 
        hdu_new = fits.PrimaryHDU(cube, hplane)
        hdu_new.verify('fix')
        hplane = hdu_new.header

    Parameter:
    ----------
    header: header_like
    nz_fin: number of frames along the 3rd axis
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
    FUNCTION TO CHANGE A HEADER WHEN THE WAVELENGTH RANGE IS REDUCED (by providing the number of channels to be removed at the beginning and at the end).
    Note: this function is different than hdifferentZ in the sense that it assumes that the cube remains a spectral cube along the z-axis. 
    """

    hplane=header.copy()
    n_z = hplane['NAXIS3']
    if n_subtr_chan_init >= hplane['CRPIX3'] or n_z-n_subtr_chan_fin < hplane['CRPIX3']:
        lbda = genlbda(header)
        hplane['CRVAL3']=lbda[n_subtr_chan_init]
        hplane['CRPIX3']=1
    else:
        hplane['CRPIX3']=hplane['CRPIX3']-n_subtr_chan_init
    nz_fin = n_z-n_subtr_chan_init-n_subtr_chan_fin
    hplane['NAXIS3']= nz_fin
    return hplane


def hdifferentZ(header,nz_fin,crpix3=1,crval3=1,cunit3='#',cdelt3=1,ctype3='cube index'):
    """
    FUNCTION TO CHANGE A HEADER WHEN THE Z DIMENSION IS CHANGED (e.g. from wavelength to cube index)
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
    IMPORTANT: IT ASSUMES THE STAR HAS BEEN PLACED ON THE CENTRAL PIXEL (via frame_center.py), and the frame has been derotated to match North up.
    """
    hplane=header.copy()
    hplane['NAXIS2']= ny_fin
    hplane['NAXIS1']= nx_fin
    hplane['CRPIX2']= int(ny_fin/2.)   #the header is 1-based (so no need of subtracting 1 to n/2)
    hplane['CRPIX1']= int(nx_fin/2.)
    hplane['CRVAL2']= header['DEC']
    hplane['CRVAL1']= header['RA']
    return hplane

