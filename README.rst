VIP - Vortex Image Processing package
=====================================


.. image:: https://badge.fury.io/py/vip-hci.svg
    :target: https://pypi.python.org/pypi/vip-hci

.. image:: https://img.shields.io/badge/Python-3.6%2C%203.7-brightgreen.svg
    :target: https://pypi.python.org/pypi/vip-hci

.. image:: https://travis-ci.org/vortex-exoplanet/VIP.svg?branch=master
    :target: https://travis-ci.org/vortex-exoplanet/VIP

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/vortex-exoplanet/VIP/blob/master/LICENSE

.. image:: https://img.shields.io/badge/arXiv-1705.06184%20-yellowgreen.svg
    :target: https://arxiv.org/abs/1705.06184

.. image:: https://readthedocs.org/projects/vip/badge/?version=latest
    :target: http://vip.readthedocs.io/en/latest/?badge=latest

.. image:: https://codecov.io/gh/vortex-exoplanet/VIP/branch/master/graph/badge.svg
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

``VIP`` is a python package for angular, reference star and spectral
differential imaging for exoplanet and disk high-contrast imaging. ``VIP`` is
compatible with Python 3 (Python 2 compatibility dropped with ``VIP`` 0.9.9).

.. image:: https://github.com/carlgogo/carlgogo.github.io/blob/master/assets/images/vip.png?raw=true
    :alt: Mosaic of S/N maps

The goal of ``VIP`` is to integrate open-source, efficient, easy-to-use and
well-documented implementations of high-contrast image processing algorithms to
the interested scientific community. The main repository of ``VIP`` resides on
`GitHub <https://github.com/vortex-exoplanet/VIP>`_, the standard for scientific
open source code distribution, using Git as a version control system.

``VIP`` started as the effort of `Carlos Alberto Gomez Gonzalez <https://carlgogo.github.io/>`_,
a former PhD student of the `VORTEX team <http://www.vortex.ulg.ac.be/>`_
(ULiege, Belgium). ``VIP``'s development is led by Dr. Gomez with contributions
made by collaborators from several teams (take a look at the `contributors tab <https://github.com/vortex-exoplanet/VIP/graphs/contributors>`_ on
``VIP``'s GitHub repository). Most of ``VIP``'s functionalities are mature but
it doesn't mean it's free from bugs. The code is continuously evolving and
therefore feedback/contributions are greatly appreciated. If you want to report
a bug or suggest a functionality please create an issue on GitHub. Pull
requests are very welcomed!


Documentation
-------------
The documentation for ``VIP`` can be found here: http://vip.readthedocs.io.


Jupyter notebook tutorial
-------------------------
Tutorials, in the form of Jupyter notebooks, showcasing ``VIP``'s usage and other resources such as test/dummy datasets are available on the ``VIP-extras`` `repository <https://github.com/carlgogo/VIP_extras>`_. Alternatively, you can execute this repository on `Binder <https://mybinder.org/v2/gh/carlgogo/VIP_extras/master>`_. The notebook for ADI processing can be visualized online with
`nbviewer <http://nbviewer.jupyter.org/github/carlgogo/VIP_extras/blob/master/tutorials/01_adi_pre-postproc_fluxpos_ccs.ipynb>`_. If you are new to the Jupyter notebook application check out the `beginner's guide
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
Homebrew/MacPorts/Fink for macOS. I personally recommend using `Miniconda <https://conda.io/miniconda>`_.

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
If you want to benefit from the ``git`` functionalities, you need to clone the
repository (make sure your system has ``git`` installed):

.. code-block:: bash

  $ git clone https://github.com/vortex-exoplanet/VIP.git

Then you can install the package by following the previous steps, using the
setup.py file. Creating a fork with GitHub is recommended to developers or to
users who want to experiment with the code.

Other dependencies
^^^^^^^^^^^^^^^^^^
``OpenCV`` (Open source Computer Vision) provides fast C++ image processing
operations and is used by ``VIP`` for basic image transformations. If you don't
have/want the ``OpenCV`` python bindings (``OpenCV`` is optional since ``VIP``
v0.5.2), ``VIP`` will use the much slower ``ndimage``/``scikit-image`` libraries
transparently. Fortunately, installing ``OpenCV`` library is nowadays and easy
process that is done automatically with the ``VIP`` installation. Alternatively,
you could use ``conda``:

.. code-block:: bash

  $ conda install opencv

``VIP`` contains a class ``vip_hci.fits.ds9`` that enables, through ``pyds9``,
the interaction with a DS9 window (displaying numpy arrays, controlling the
display options, etc). ``pyds9`` is an optional requirement and must be
installed from the latest development version:

.. code-block:: bash

    $ pip install git+git://github.com/ericmandel/pyds9.git#egg=pyds9

Also, you can install the Intel Math Kernel Library (MKL) optimizations
(provided that you have a recent version of ``conda``) or ``openblas``
libraries. Either of them can be installed with ``conda install``. This is
recommended along with ``OpenCV`` for maximum speed on ``VIP`` computations.

``VIP`` offers the possibility of computing SVDs on GPU by using ``CuPy``
(starting from version 0.8.0) or ``PyTorch`` (from version 0.9.2). These remain
as optional requirements, to be installed by the user, as well as a proper CUDA
environment (and a decent GPU card).

Loading VIP
^^^^^^^^^^^
Finally, start Python (or IPython or a Jupyter notebook if you prefer) and check
that you are able to import ``VIP``:

.. code-block:: python

  import vip_hci as vip

If everything went fine with the installation, you will see a welcome message.
Now you can start finding exoplanets!


Mailing list
------------
Please subscribe to our `mailing list <http://lists.astro.caltech.edu:88/mailman/listinfo/vip>`_
if you want to be informed of ``VIP``'s latest developments (new versions
and/or updates).


Attribution
-----------
Please cite Gomez Gonzalez et al. 2017 (http://iopscience.iop.org/article/10.3847/1538-3881/aa73d7/)
whenever you publish data reduced with ``VIP``. Astrophysics Source Code Library
reference [ascl:1603.003].

