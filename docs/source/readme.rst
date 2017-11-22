VIP - Vortex Image Processing package
=====================================


.. image:: https://badge.fury.io/py/vip-hci.svg
    :target: https://pypi.python.org/pypi/vip-hci

.. image:: https://img.shields.io/badge/Python-2.7%2C%203.6-brightgreen.svg
    :target: https://pypi.python.org/pypi/vip-hci

.. image:: https://travis-ci.org/vortex-exoplanet/VIP.svg?branch=master
    :target: https://travis-ci.org/vortex-exoplanet/VIP

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/vortex-exoplanet/VIP/blob/master/LICENSE

.. image:: https://img.shields.io/badge/arXiv-1705.06184%20-yellowgreen.svg
    :target: https://arxiv.org/abs/1705.06184

.. image:: https://readthedocs.org/projects/vip/badge/?version=latest
    :target: http://vip.readthedocs.io/en/latest/?badge=latest


Introduction
------------

``VIP`` is a python package for angular, reference star and spectral
differential imaging for exoplanet/disk detection through high-contrast imaging.
``VIP`` is compatible with Python 2.7 and 3.6.

The goal of ``VIP`` is to incorporate open-source, efficient, easy-to-use and
well-documented implementations of high-contrast image processing algorithms to
the interested scientific community. The main repository of ``VIP`` resides on
`Github <https://github.com/vortex-exoplanet/VIP>`_, the standard for scientific
open source code distribution, using Git as a version control system.

``VIP`` started as the effort of a `PhD student <https://github.com/carlgogo/>`_
within the `VORTEX team <http://www.vortex.ulg.ac.be/>`_ (Liege, Belgium). It is
currently developed by collaborators from several institutes/teams. The tab
contributors on ``VIP``'s Github repo shows the direct contributions to the code.
Most of ``VIP``'s functionalities are mature but it doesn't mean it's free from
bugs. The code is continuously evolving and therefore feedback/contributions are
greatly appreciated. If you want to report a bug, suggest or add a functionality
please create an issue or send a pull request on
`the repository <https://github.com/vortex-exoplanet/VIP>`_.


Documentation
-------------
The documentation for ``VIP`` can be found here: http://vip.readthedocs.io.


Jupyter notebook tutorial
-------------------------
A tutorial (Jupyter notebook) showing he use of ``VIP`` for ADI data processing
is available in `this repository <https://github.com/carlgogo/vip-tutorial>`_.
Alternatively, it can be visualized online
`here <http://nbviewer.jupyter.org/github/carlgogo/vip-tutorial/blob/master/Tutorial1_VIP_adi_pre-postproc_fluxpos_ccs.ipynb>`_.
If you are new to the Jupyter notebook application check out the `beginner guide
<https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html>`_.


TL;DR Guide
-----------------
.. code-block:: bash

    $ pip install vip_hci


Installation and dependencies
-----------------------------
The benefits of using a Python package manager (distribution), such as Anaconda or
Canopy, are many. Mainly, it brings easy and robust package management and
avoids messing up with your system's default python. An alternative is to use
package managers like apt-get for Ubuntu or
Homebrew/MacPorts/Fink for Macos. I personally recommend using Miniconda which you
can find here: https://conda.io/miniconda.html.

``VIP`` depends on existing packages from the Python ecosystem, such as
``numpy``, ``scipy``, ``matplotlib``, ``pandas``, ``astropy``, ``scikit-learn``,
``scikit-image``, ``photutils`` and ``emcee``. There are different ways of
installing ``VIP`` suitable for different scenarios.


Using PIP
^^^^^^^^^
The easiest way to install ``VIP`` is through the Python Package Index, aka
`Pypi <https://pypi.org/>`_, with the ``pip`` package manager. Simply run:

.. code-block:: bash

  $ pip install vip_hci

With ``pip`` you can easily uninstall, upgrade or install a specific version of
``VIP``. For upgrading the package run:

.. code-block:: bash

  $ pip install --upgrade vip_hci

Alternatively, you can use ``pip install`` and point to the Github repo:

.. code-block:: bash

  $ pip install git+https://github.com/vortex-exoplanet/VIP.git

Using the setup.py file
^^^^^^^^^^^^^^^^^^^^^^^
You can download ``VIP`` from its Github repository as a zip file. A setup.py
file (Setuptools) is included in the root folder of ``VIP``. Enter the package's
root folder and run:

.. code-block:: bash

  $ python setup.py install

Using GIT
^^^^^^^^^
If you want to benefit from the ``git`` functionalities, you need to clone the
repository (make sure your system has ``git`` installed):

.. code-block:: bash

  $ git clone https://github.com/vortex-exoplanet/VIP.git

Then you can install the package by following the previous steps, using the
setup.py file. Creating a fork with Github is recommended to developers or to
users who want to experiment with the code.

Other dependencies
^^^^^^^^^^^^^^^^^^
``Opencv`` (Open source Computer Vision) provides fast c++ image processing
operations and is used by ``VIP`` for basic image transformations. If you don't
have/want the ``opencv`` python bindings (``opencv`` is optional since ``VIP``
v0.5.2), ``VIP`` will use the much slower ``ndimage/scikit-image`` libraries
transparently. Fortunately, installing ``opencv`` library is nowadays and easy
process that is done automatically with the ``VIP`` installation. Alternatively,
you could use ``conda``:

.. code-block:: bash

  $ conda install opencv

``VIP`` ships a stripped-down version of ``RO.DS9`` (by Russell Owen) for convenient
``xpaset/xpaget`` based interaction with ``DS9``. ``VIP`` contains a class
``vipDS9`` that works on top of ``RO.DS9`` containing several useful methods for
``DS9`` control such as displaying arrays, manipulating regions, controlling the
display options, etc. ``VipDS9`` functionality will only be available if you have
``DS9`` and ``XPA`` installed on your system PATH.

Also, optionally you can install the Intel Math Kernel Library (MKL)
optimizations provided that you have Anaconda(>v2.5) and ``conda`` on your
system. This is recommended along with ``Opencv`` for maximum speed on ``VIP``
computations. Run:

.. code-block:: bash

  $ conda install mkl

Starting from version 0.8.0 ``VIP`` offers the possibility of computing SVDs
on GPU by using ``cupy``. This remains an optional requirement, to be installed
by the user, as it requires having a real (good) GPU card and a proper CUDA
environment.

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
You can subscribe to our `mailing <http://lists.astro.caltech.edu:88/mailman/listinfo/vip>`_
list if you want to be informed of the latest developments of the ``VIP`` package
(new versions and/or updates).


Attribution
-----------
Please cite Gomez Gonzalez et al. 2017 (http://iopscience.iop.org/article/10.3847/1538-3881/aa73d7/)
whenever you publish data reduced with ``VIP``. Astrophysics Source Code Library
reference [ascl:1603.003].

