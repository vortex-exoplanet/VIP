VIP - Vortex Image Processing package
=====================================


.. image:: https://badge.fury.io/py/vip-hci.svg
    :target: https://pypi.python.org/pypi/vip-hci

.. image:: https://img.shields.io/badge/Python-3.7%2C%203.8%2C%203.9-brightgreen.svg
    :target: https://pypi.python.org/pypi/vip-hci

.. image:: https://travis-ci.com/vortex-exoplanet/VIP.svg?branch=master
    :target: https://travis-ci.com/vortex-exoplanet/VIP

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/vortex-exoplanet/VIP/blob/master/LICENSE

.. image:: https://img.shields.io/badge/arXiv-1705.06184%20-yellowgreen.svg
    :target: https://arxiv.org/abs/1705.06184

.. image:: https://readthedocs.org/projects/vip/badge/?version=latest
    :target: http://vip.readthedocs.io/en/latest/?badge=latest

.. image:: https://codecov.io/gh/vortex-exoplanet/VIP/branch/master/graph/badge.svg?token=HydCFQqLRf
    :target: https://codecov.io/gh/vortex-exoplanet/VIP
    

::

    ---------------------------------------------------
            oooooo     oooo ooooo ooooooooo.
             `888.     .8'  `888' `888   `Y88.
              `888.   .8'    888   888   .d88'
               `888. .8'     888   888ooo88P'
                `888.8'      888   888
                 `888'       888   888
                  `8'       o888o o888o
    ---------------------------------------------------
             Vortex Image Processing package
    ---------------------------------------------------


Introduction
------------

``VIP`` is a python package for high-contrast imaging of exoplanets and circumstellar disks. 
``VIP`` is compatible with Python 3.7, 3.8 and 3.9 (Python 2 compatibility dropped with ``VIP`` 0.9.9).

The goal of ``VIP`` is to integrate open-source, efficient, easy-to-use and
well-documented implementations of high-contrast image processing algorithms to
the interested scientific community. The main repository of ``VIP`` resides on
`GitHub <https://github.com/vortex-exoplanet/VIP>`_, the standard for scientific
open source code distribution, using Git as a version control system.

Most of ``VIP``'s functionalities are mature but
it does not mean it is free from bugs. The code is continuously evolving and
therefore feedback/contributions are greatly appreciated. Please refer to `these instructions <https://vip.readthedocs.io/en/latest/Contact.html>`_ if you want to report
a bug, ask a question, suggest a new functionality or contribute to the code (the latter is particularly welcome)!

.. image:: https://github.com/carlgogo/carlgogo.github.io/blob/master/assets/images/vip.png?raw=true
    :alt: Mosaic of S/N maps

Documentation
-------------
The documentation for ``VIP`` can be found here: http://vip.readthedocs.io.


Jupyter notebook tutorial
-------------------------
Tutorials, in the form of Jupyter notebooks, showcasing ``VIP``'s usage and 
other resources such as test datasets are available in the 
``VIP-extras`` `repository <https://github.com/vortex-exoplanet/VIP_extras>`_. 
Alternatively, you can execute this repository on 
`Binder <https://mybinder.org/v2/gh/vortex-exoplanet/VIP_extras/master>`_ (in the tutorials directory). The first (quick-start) notebook can be visualized online with
`nbviewer <http://nbviewer.jupyter.org/github/vortex-exoplanet/VIP_extras/blob/master/tutorials/01_quickstart.ipynb>`_. 
If you are new to the Jupyter notebook application check out the `beginner's guide
<https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html>`_.


TL;DR setup guide
-----------------
.. code-block:: bash

    $ pip install vip_hci


Installation and dependencies
-----------------------------
The benefits of using a Python package manager (distribution), such as
(ana)conda or Canopy, are many. Mainly, it brings easy and robust package
management and avoids messing up with your system's default python. An
alternative is to use package managers like apt-get for Ubuntu or
Homebrew/MacPorts/Fink for macOS. We recommend using 
`Miniconda <https://conda.io/miniconda>`_.

``VIP`` depends on existing packages from the Python ecosystem, such as
``numpy``, ``scipy``, ``matplotlib``, ``pandas``, ``astropy``, ``scikit-learn``,
``scikit-image``, ``photutils`` and others. There are different ways of
installing ``VIP`` suitable for different scenarios.


Using pip
^^^^^^^^^
The easiest way to install ``VIP`` is through the Python Package Index, aka
`PyPI <https://pypi.org/>`_, with the ``pip`` package manager. Simply run:

.. code-block:: bash

  $ pip install vip_hci

With ``pip`` you can easily uninstall, upgrade or install a specific version of
``VIP``. For upgrading the package run:

.. code-block:: bash

  $ pip install --upgrade vip_hci

Alternatively, you can use ``pip install`` and point to the GitHub repo:

.. code-block:: bash

  $ pip install git+https://github.com/vortex-exoplanet/VIP.git

Using the setup.py file
^^^^^^^^^^^^^^^^^^^^^^^
You can download ``VIP`` from its GitHub repository as a zip file. A ``setup.py``
file (setuptools) is included in the root folder of ``VIP``. Enter the package's
root folder and run:

.. code-block:: bash

  $ python setup.py install


Using Git
^^^^^^^^^
If you plan to contribute or experiment with the code you need to make a 
fork of the repository (click on the fork button in the top right corner) and 
clone it:

.. code-block:: bash

  $ git clone https://github.com/<replace-by-your-username>/VIP.git

If you do not create a fork, you can still benefit from the ``git`` syncing
functionalities by cloning the repository (but will not be able to contribute):

.. code-block:: bash

  $ git clone https://github.com/vortex-exoplanet/VIP.git

Before installing the package, it is highly recommended to create a dedicated
conda environment to not mess up with the package versions in your base 
environment. This can be done easily with (replace vipenv by the name you want
for your environment):

.. code-block:: bash

  $ conda create -n vipenv python=3.9 ipython

Note: installing ipython while creating the environment with the above line will
avoid a commonly reported issue which stems from trying to import VIP from 
within a base python2.7 ipython console.

To install VIP, simply cd into the VIP directory and run the setup file 
in 'develop' mode:

.. code-block:: bash

  $ cd VIP
  $ python setup.py develop

If cloned from your fork, make sure to link your VIP directory to the upstream 
source, to be able to easily update your local copy when a new version comes 
out or a bug is fixed:

.. code-block:: bash

  $ git add remote upstream https://github.com/vortex-exoplanet/VIP.git

If you plan to develop VIP or use it intensively, it is highly recommended to 
also install the optional dependencies listed below.


Optional dependencies
^^^^^^^^^^^^^^^^^^^^^
The following dependencies are not automatically installed upon installation of ``VIP`` but may significantly improve your experience:

- ``VIP`` contains a class ``vip_hci.vip_ds9`` that enables, through ``pyds9``, the interaction with a DS9 window (displaying numpy arrays, controlling the display options, etc). To enable this feature, ``pyds9`` must be installed from the latest development version: ``pip install git+git://github.com/ericmandel/pyds9.git#egg=pyds9``
- Also, you can install the Intel Math Kernel Library (``mkl``) optimizations (provided that you have a recent version of ``conda``) or ``openblas`` libraries. Either of them can be installed with ``conda install``. 
- ``VIP`` offers the possibility of computing SVDs on GPU by using ``CuPy`` (starting from version 0.8.0) or ``PyTorch`` (from version 0.9.2). These remain as optional requirements, to be installed by the user, as well as a proper CUDA environment (and a decent GPU card).
- Finally, bad pixel correction routines can be optimised with ``Numba``, which  converts some Python code, particularly ``NumPy``, into fast machine code. A factor up to ~50x times speed improvement can be obtained on large images compared to NumPy. Numba can be installed with ``conda install numba``.


Loading VIP
^^^^^^^^^^^
Finally, start Python (or IPython or a Jupyter notebook if you prefer) and check
that you are able to import ``VIP``:

.. code-block:: python

  import vip_hci as vip

If everything went fine with the installation, you will see a welcome message.
Now you can start finding exoplanets!


Image conventions
-----------------

By default, VIP routines are compatible with either even- or odd-dimension input frames. For VIP routines that require the star to be centered in the input images (e.g. post-processing routines involving (de)rotation or scaling), the code will assume that it is placed on (zero-based indexing):

- size/2-0.5 for odd-size input images; 
- size/2 for even-size input images;

i.e. exactly on a pixel in either cases. The VIP recentering routines will place the star centroid at one of these locations accordingly.


Contact
-------
Answers to `frequently asked questions <https://vip.readthedocs.io/en/latest/faq.html>`_ are provided in the relevant section of the documentation.
If you have an issue with VIP, please first check it is not detailed in the FAQ.
If you find a bug or experience an unreported issue in VIP, it is recommended to post a new entry in the `Issues section <https://github.com/vortex-exoplanet/VIP/issues>`_ on GitHub. Feel free to propose a pull request if you have already identified the source of the bug/issue.

If you have a global comment, inquiry about how to solve a specific task using VIP, or suggestions to improve VIP, feel free to open a new thread in the `Discussions <https://github.com/vortex-exoplanet/VIP/discussions>`_ section. The 'Discussions' section is also used to post VIP-related announcements and discuss recent/on-going changes in VIP.
Envisioned future developments are listed in the `Projects <https://github.com/vortex-exoplanet/VIP/projects/1>`_ section. Contributions are very welcome!

If you wish to be kept informed about major VIP updates and on-going/future developments, feel free to click the 'watch' button at the top of the GitHub page.


Attribution
-----------

``VIP`` started as the effort of `Carlos Alberto Gomez Gonzalez <https://github.com/carlos-gg>`_,
a former PhD student of `PSILab <https://sites.google.com/site/olivierabsil/psilab>`_
(ULiege, Belgium), who has led the development of VIP from 2015 to 2020.
Maintenance and current development is now led by `Valentin Christiaens <https://github.com/VChristiaens>`_.
VIP benefitted from contributions made by collaborators from several teams, including: Ralf Farkas, Julien Milli, Olivier Wertz, Henry Ngo, Alan Rainot, Gary Ruane, Corentin Doco, Miles Lucas, Gilles Orban de Xivry, Lewis Picker, Faustine Cantalloube, Iain Hammond, Christian Delacroix, Arthur Vigan, Dimitri Mawet and Olivier Absil.
More details about the respective contributions are available `here <https://github.com/vortex-exoplanet/VIP/graphs/contributors?from=2015-07-26&to=2022-03-29&type=a>`_.

Please cite `Gomez Gonzalez et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017AJ....154....7G/abstract>`_ whenever 
you publish data reduced with ``VIP`` (Astrophysics Source Code Library reference `ascl:1603.003`).
In addition, please cite the relevant publication(s) for the algorithms you use within VIP (usually mentioned in the documentation, e.g. `Marois et al. 2006 <https://ui.adsabs.harvard.edu/abs/2006ApJ...641..556M/abstract>`_ for median-ADI).

Note: A new JOSS paper led by Valentin Christiaens is in preparation.
