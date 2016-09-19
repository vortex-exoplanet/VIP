Installation and dependencies
------------------------------
The benefits of a python package manager or distribution, such as Canopy or
Anaconda, are multiple. Mainly it brings easy and robust package management and
avoids messing up with your system's default python. An alternative is to use
package managers like apt-get for Ubuntu or
Homebrew/MacPorts/Fink for OSX. I personally recommend using Anaconda which you
can find here: https://www.continuum.io/downloads.

A setup.py file (Setuptools Python package) is included in the root folder of
VIP. It takes care of installing most of the requirements for you. VIP depends on
existing packages from the Python ecosystem, such as numpy, scipy, matplotlib,
pandas, astropy, scikit-learn, scikit-image, photutils, emcee, etc.

VIP ships a stripped-down version of RO.DS9 (Russell Owen) for convenient
xpaset/xpaget based interaction with DS9. VIP contains a class vipDS9 that works
on top of RO.DS9 containing several useful methods for DS9 control such as
displaying arrays, manipulating regions, controlling the display options, etc.
VipDS9 functionality will only be available if you have DS9 and XPA installed
on your system PATH.

OpenCV (Open source Computer Vision) is used for fast image processing operations.
Opencv and its python bindings are usually difficult to install (can't be processed
from the setup.py requirements) but fortunately with just one command we'll get it
ready.

Make sure to install OpenCV before running setup.py and make sure you have a C
compiler installed in your system, like g++. With conda (anaconda) just type:

.. code-block:: bash

  $ conda install opencv

For installing VIP in your system run setup.py:

.. code-block:: bash

  $ python setup.py install

The code is in continuous development and will be changing often. It's preferred
to 'install' with the develop flag:

.. code-block:: bash

  $ python setup.py develop

In any case wait a couple of minutes until all the requirements are satisfied.
