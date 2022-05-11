---
title: 'VIP: A Python package for high-contrast imaging'
tags:
  - Python
  - astronomy
  - exoplanets
  - high-contrast
  - direct imaging
authors:
  - name: Valentin Christiaens
    orcid: 0000-0002-0101-8814
    affiliation: 1
  - name: Carlos Alberto Gómez Gonzalez
    orcid: 0000-0003-2050-1710
    affiliation: 2
  - name: Ralf Farkas
    orcid: 0000-0002-7647-1429
    affiliation: 3
  - name: Carl-Henrik Dahlqvist
    orcid: 0000-0003-4994-9244
    affiliation: 1
  - name: Evert Nasedkin
    orcid: 0000-0002-9792-3121
    affiliation: 4
  - name: Julien Milli
    orcid: 0000-0001-9325-2511
    affiliation: 5
  - name: Olivier Absil
    orcid: 0000-0002-4006-6237
    affiliation: 1
  - name: Henry Ngo
    orcid: 0000-0001-5172-4859
    affiliation: 6
  - name: Carles Cantero
    orcid: 0000-0003-2073-782X
    affiliation: 1,7
  - name: Alan Rainot
    orcid: 0000-0001-9904-0624
    affiliation: 8
  - name: Iain Hammond
    orcid: 0000-0003-1502-4315
    affiliation: 9
  - name: Arthur Vigan
    orcid: 0000-0002-5902-7828
    affiliation: 10

affiliations:
  - name: Space sciences, Technologies & Astrophysics Research Institute, Université de Liège, Belgium
    index: 1
  - name: Barcelona Supercomputing Center, Barcelona, Spain
    index: 2
  - name: Rheinische Friedrich-Wilhelms-Universität Bonn, Germany
    index: 3
  - name: Max-Planck-Institut für Astronomie, Heidelberg, Germany
    index: 4
  - name: Univ. Grenoble Alpes, CNRS, IPAG, F-38000 Grenoble, France
    index: 5
  - name: NRC Herzberg Astronomy and Astrophysics, Victoria, BC, Canada
    index: 6
  - name: Montefiore Institute, Université de Liège, 4000 Liège, Belgium
    index: 7
  - name: Institute of Astronomy, KU Leuven, Belgium
    index: 8
  - name: School of Physics and Astronomy, Monash University, Vic 3800, Australia
    index: 9
  - name: Aix Marseille Univ, CNRS, CNES, LAM, Marseille, France
    index: 10

date: 4 May 2022
bibliography: paper.bib
---

# Summary

Direct imaging of exoplanets and circumstellar disks at optical and infrared
wavelengths requires reaching high contrasts at short angular separations. This
can only be achieved through the synergy of advanced instrumentation, such as
adaptive optics and coronagraphy, with a relevant combination of observing strategy
and post-processing algorithms to model and subtract residual starlight. In this
context, ``VIP`` is a Python package providing the tools to reduce,
post-process and analyze high-contrast imaging datasets, enabling the detection
and characterization of directly imaged exoplanets, circumstellar disks, and
stellar environments.

# Statement of need

``VIP`` stands for Vortex Image Processing. It is a collaborative project
which started at the University of Liège, aiming to integrate open-source,
efficient, easy-to-use and well-documented implementations of state-of-the-art
algorithms used in the context of high-contrast imaging. The package follows a
modular architecture, such that its routines cover a wide diversity of tasks,
including:

* image pre-processing, such as sky subtraction, bad pixel correction, bad
frame removal, or image alignment and star centering (`preproc` module);

* modeling and subtracting the stellar point spread function (PSF) using 
state-of-the-art algorithms that leverage observing strategies such as angular 
differential imaging  (ADI), spectral differential imaging (SDI) or reference 
star differential imaging [@Marois:2006; @Sparks:2002; @Ruane:2019], which 
induce diversity between speckle and authentic astrophysical signals (`psfsub` 
module);

* characterizing point sources and extended circumstellar signals through
forward modeling (`fm` module);

* detecting and characterizing point sources through inverse approaches
(`invprob` module);

* assessing the achieved contrast in PSF-subtracted images, automatically
detecting point sources, and estimating their significance (`metrics` module).

The features implemented in ``VIP`` as of 2017 are described in @Gomez:2017. 
Since then, the package has been widely used by the high-contrast imaging 
community for the discovery of low-mass companions 
[@Milli:2017;  @Hirsch:2019;  @Ubeira:2020], their characterization 
[@Wertz:2017;  @Delorme:2017;  @Christiaens:2018;  @Christiaens:2019], the study 
of planet formation [@Ruane:2017;  @Reggiani:2018;  @Mauco:2020;  @Toci:2020], 
the study of high-mass star formation [@Rainot:2020;  @Rainot:2022] ,the study 
of debris disks [@Milli:2017b; @Milli:2019], or the development of new 
high-contrast imaging algorithms 
[@Gomez:2018;  @Dahlqvist:2020;  @Pairet:2021;  @Dahlqvist:2021]. Given the 
rapid expansion of ``VIP``, we summarize here all novelties that were brought 
to the package over the past five years.

The rest of this manuscript summarizes all major changes since v0.7.0
[@Gomez:2017], that are included in the latest release of ``VIP`` (v1.3.0). At
a structural level, ``VIP`` underwent a major change since version v1.1.0, which
aimed to migrate towards a more streamlined and easy-to-use architecture. The
package now revolves around five major modules (`fm`, `invprob`, `metrics`,
`preproc` and `psfsub`, as described above) complemented by four additional
modules containing various utility functions (`config`, `fits`,
`stats` and `var`). New `Dataset` and `Frame` classes have also been
implemented, enabling an object-oriented approach for processing high-contrast
imaging datasets and analyzing final images, respectively. Similarly, a
`HCIPostProcAlgo` class and different subclasses inheriting from it have been
defined to facilitate an object-oriented use of ``VIP`` routines.

Some of the major changes in each module of ``VIP`` are summarized below:

* `fm`:
    - new routines were added to create parametrizable scattered-light disk
    models and extended signals in ADI cubes, in order to forward-model the
    effect of ADI post-processing [@Milli:2012; @Christiaens:2019];
    - the log-likelihood expression used in the negative fake companion (NEGFC)
    technique was updated, and the default convergence criterion for the
    NEGFC-MCMC method is now based on auto-correlation [@Christiaens:2021];
    - the NEGFC methods are now fully compatible with integral field
    spectrograph (IFS) input datacubes.

* `invprob`:
    - a Python implementation of the ANDROMEDA algorithm [@Cantalloube:2015] is
    now available as part of ``VIP``;
    - the KLIP-FMMF and LOCI-FMMF algorithms
    [@Pueyo:2016; @Ruffio:2017; @Dahlqvist:2021] are now also available in the 
    `invprob` module.
    - a Python implementation of the PACO algorithm [@Flasseur:2018] is now
    also available, including both the planet detection and flux estimation 
    algorithms.

* `metrics`:
    - calculation of standardized trajectory maps (STIM) is now available
    [@Pairet:2019];
    - functions to calculate completeness-based contrast curves and completeness 
    maps, inspired by the framework in @JensenClem:2018 and implemented as in 
    @Dahlqvist:2021, have now been added to the `metrics` module.

* `preproc`:
    - the module now boasts several new algorithms for (i) the identification
    of either isolated bad pixels or clumps of bad pixels, leveraging on
    iterative sigma filtering (`cube_fix_badpix_clump`), the circular symmetry
    of the PSF (`cube_fix_badpix_annuli`), or the radial expansion of the PSF
    with increasing wavelength (`cube_fix_badpix_ifs`), and (ii) the correction
    of bad pixels based on either median replacement (default) or Gaussian
    kernel interpolation (`cube_fix_badpix_with_kernel`);
    - a new algorithm was added for the recentering of coronagraphic image cubes
    based on the cross-correlation of the speckle pattern, after appropriate
    filtering and log-scaling of pixel intensities [@Ruane:2019].

* `psfsub`:
    - all principal component analysis (PCA) based routines
    [@Amara:2012; @Soummer:2012] have been re-written for improved efficiency,
    and are now also compatible with 4D IFS+ADI input cubes to apply SDI-based
    PSF modeling and subtraction algorithms;
    - an implementation of the Locally Optimal Combination of Images algorithm
    [@Lafreniere:2007] was added;
    - an annular version of the non-negative matrix factorization algorithm
    is now available [@Lee:1999; @Gomez:2017];
    - besides median-ADI, the `medsub` routine now also supports median-SDI.

We refer the interested reader to release descriptions and GitHub
[announcements](https://github.com/vortex-exoplanet/VIP/discussions/categories/announcements)
for a more complete list of all changes, including improvements not mentioned
in the above summary.

Two major convention updates are also to be noted in ``VIP``. All image
operations (rotation, scaling, resampling and sub-pixel shifts) are now
performed using Fourier-Transform (FT) based methods by default. These have
been implemented as low-level routines in the `preproc` module. FT-based methods
significantly outperform interpolation-based methods in terms of flux
conservation [@Larkin:1997]. However, given the order of magnitude slower
computation of FT-based image rotations, the option to use interpolation-based
methods is still available in all relevant ``VIP`` functions. The second change
of convention concerns the assumed center for even-size images, which is now
defined as the top-right pixel among the four central pixels of the image - a
change motivated by the new default FT-based methods for image operations. The
center convention is unchanged for odd-size images (central pixel).

Finally, a total of nine jupyter notebook tutorials covering most of the
available features in VIP were implemented. These tutorials illustrate how to
(i) load and post-process an ADI dataset (quick-start tutorial); (ii)
pre-process ADI and IFS datasets; (iii) model and subtract the stellar halo with
ADI-based algorithms; (iv) calculate metrics such as the S/N ratio
[@Mawet:2014], STIM maps [@Pairet:2019] and contrast curves; (v) find the radial
separation, azimuth and flux of a point source; (vi) create and forward model
scattered-light disk models; (vii) post-process IFS data and infer the exact
astro- and photometry of a given point source; (viii) use FT-based and
interpolation-based methods for different image operations, and assess their
respective performance; and (ix) use the new object-oriented framework for
``VIP``.

# Acknowledgements

An up-to-date list of contributors to VIP is available
[here](https://github.com/vortex-exoplanet/VIP/graphs/contributors?from=2015-07-26&to=2022-04-27&type=a).
VC acknowledges financial support from the Belgian F.R.S.-FNRS. This project
has received funding from the European Research Council (ERC) under the
European Union’s FP7 and Horizon 2020 research and innovation programmes (grant
agreements No 337569 and 819155), and from the Wallonia-Brussels Federation
(grant for Concerted Research Actions).

# References
