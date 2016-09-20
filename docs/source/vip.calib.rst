vip.calib package
=================

The subpackage ``calib`` contains some useful cosmetics and pre-processing
functionalities:
    - resizing frames and cubes : upscaling/pixel binning,
    - shifting frames,
    - rotating frames and cubes,
    - cropping frames and cubes,
    - removing bad pixels from frames and cubes,
    - correcting nan values from frames and cubes,
    - detecting bad frames in cubes, using:
        - pixel statistics in annulus or circular aperture,
        - ellipticity of a point like source,
        - frames correlation,
    - temporal subsampling of cubes (mean, median, trimmed mean),
    - registration (re-centering) of frames, using:
        - centroid fitting a 2d gaussian or moffat,
        - DFT upsampling or fourier cross-correlation (Guizar et al. 2008),
        - radon transform for broadband frames (Pueyo et al. 2014),
        - using satellite/waffle spots (fitting plus intersection).

Astronomical calibration functionality like flat fielding and dark-sky
subtraction, in spite of its simplicity was not included in VIP because of the
heterogeneity of the datasets coming from different observatories (each having
different data storage and headers). You can perform this in python in procedures
of a few lines or using dedicated instrument pipelines such as esorex (ESO
instruments).



Submodules
----------

vip.calib.badframes module
--------------------------

.. automodule:: vip.calib.badframes
    :members:
    :undoc-members:
    :show-inheritance:

vip.calib.badpixremoval module
------------------------------

.. automodule:: vip.calib.badpixremoval
    :members:
    :undoc-members:
    :show-inheritance:

vip.calib.cosmetics module
--------------------------

.. automodule:: vip.calib.cosmetics
    :members:
    :undoc-members:
    :show-inheritance:

vip.calib.cosmetics_ifs module
------------------------------

.. automodule:: vip.calib.cosmetics_ifs
    :members:
    :undoc-members:
    :show-inheritance:

vip.calib.derotation module
---------------------------

.. automodule:: vip.calib.derotation
    :members:
    :undoc-members:
    :show-inheritance:

vip.calib.parangles module
--------------------------

.. automodule:: vip.calib.parangles
    :members:
    :undoc-members:
    :show-inheritance:

vip.calib.recentering module
----------------------------

.. automodule:: vip.calib.recentering
    :members:
    :undoc-members:
    :show-inheritance:

vip.calib.rescaling module
--------------------------

.. automodule:: vip.calib.rescaling
    :members:
    :undoc-members:
    :show-inheritance:

vip.calib.subsampling module
----------------------------

.. automodule:: vip.calib.subsampling
    :members:
    :undoc-members:
    :show-inheritance:


.. automodule:: vip.calib
    :members:
    :undoc-members:
    :show-inheritance:
