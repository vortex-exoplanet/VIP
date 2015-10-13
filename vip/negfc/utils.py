#! /usr/bin/env python

"""
Module with post-processing related functions called from within the NFC
algorithm.
"""

import numpy as np
import math
from matplotlib.pyplot import plot, xlim, ylim, hold, axes, gca, show


def radial_to_eq(r=1, t=0, rError=0, tError=0, display=False):
    """ 
    Convert the position given in (r,t) into \delta RA and \delta DEC, as 
    well as the corresponding uncertainties. 
    t = 0 deg (resp. 90 deg) points toward North (resp. East).   

    Parameters
    ----------
    r: float
        The radial coordinate.
    t: float
        The angular coordinate.
    rError: float
        The error bar related to r.
    tError: float
        The error bar related to t.
    display: boolean, optional
        If True, a figure illustrating the error ellipse is displayed.
        
    Returns
    -------
    out : tuple
        ((RA, RA error), (DEC, DEC error))
                              
    """  
    ra = (r * np.sin(math.radians(t)))
    dec = (r * np.cos(math.radians(t)))   
    u, v = (ra, dec)
    
    nu = np.mod(np.pi/2.-math.radians(t), 2*np.pi)
    a, b = (rError,r*np.sin(math.radians(tError)))

    beta = np.linspace(0,2*np.pi,5000)
    x, y = (u + (a * np.cos(beta) * np.cos(nu) - b * np.sin(beta) * np.sin(nu)),
            v + (b * np.sin(beta) * np.cos(nu) + a * np.cos(beta) * np.sin(nu)))
    
    raErrorInf = u - np.amin(x)
    raErrorSup = np.amax(x) - u
    decErrorInf = v - np.amin(y)
    decErrorSup = np.amax(y) - v        

    if display:        
        hold(True)
        plot(u,v,'ks',x,y,'r')        
        plot((r+rError) * np.cos(nu), (r+rError) * np.sin(nu),'ob',
             (r-rError) * np.cos(nu), (r-rError) * np.sin(nu),'ob')
        plot(r * np.cos(nu+math.radians(tError)), 
             r*np.sin(nu+math.radians(tError)),'ok')
        plot(r * np.cos(nu-math.radians(tError)), 
             r*np.sin(nu-math.radians(tError)),'ok')
        plot(0,0,'og',np.cos(np.linspace(0,2*np.pi,10000)) * r, 
             np.sin(np.linspace(0,2*np.pi,10000)) * r,'y')
        plot([0,r*np.cos(nu+math.radians(tError*0))],
             [0,r*np.sin(nu+math.radians(tError*0))],'k')
        axes().set_aspect('equal')
        lim = np.amax([a,b]) * 2.
        xlim([ra-lim,ra+lim])
        ylim([dec-lim,dec+lim])
        gca().invert_xaxis()
        show()
        
    return ((ra,np.mean([raErrorInf,raErrorSup])),
            (dec,np.mean([decErrorInf,decErrorSup])))    
    
    
def index_to_polar(y, x, ceny=0, cenx=0):
    """
    Convert pixel index into polar coordinates (r,theta) with 
    respect to a given center (cenx,ceny).
    
    Note that ds9 index (x,y) = Python matrix index (y,x).
    
    Parameters
    ----------
    x,y: float
        The pixel index
        
    Returns
    -------
    
    out : tuple
        The polar coordinates (r,theta) with respect to the (cenx,ceny). 
        Note that theta is given in degrees.
        
    """
    r = np.sqrt((y-ceny)**2 + (x-cenx)**2)
    theta = np.degrees(np.arctan2(y-ceny, x-cenx)) 
    
    return (r,theta)       

