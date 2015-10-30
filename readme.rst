------------------------------------
 oooooo     oooo ooooo ooooooooo.   
  `888.     .8'  `888' `888   `Y88. 
   `888.   .8'    888   888   .d88' 
    `888. .8'     888   888ooo88P'  
     `888.8'      888   888         
      `888'       888   888         
       `8'       o888o o888o
------------------------------------
  Vortex Image Processing pipeline
------------------------------------

VIP is a package/pipeline for angular, reference star and spectral differential
imaging for exoplanet detection through high-contrast imaging. VIP is being
developed in Python 2.7.

VIP is being developed within the VORTEX team @ University of Liege (Belgium).
It's in alpha version meaning that the code will change drastically before the
first release version. If you want to report a bug, suggest a feature or add a 
feature please contact the main developer at cgomez [at] ulg.ac.be or through 
github.

Please cite Gomez Gonzalez et al. 2016 (in prep.) whenever you publish data 
reduced with VIP.


DOCUMENTATION
=============
In ./tutorial folder you can find a Jupyter notebook with a detailed tutorial
of VIP. You can visualize it, without loading it locally, here:
http://nbviewer.ipython.org/github/vortex-exoplanet/VIP/blob/master/tutorial/Tutorial_VIP.ipynb

Docstrings (internal documentation) are filled in for every function in VIP.
Sphinx can be used to generate the documentation from the docstrings in html or
pdf into docs folder (Coming soon).


QUICK INSTRUCTIONS
==================
Install opencv. If you use Anaconda run:
$ conda install opencv
From the root of the VIP package:
$ python setup.py develop   
That's it!


DEPENDENCIES
============
You must have a python distribution installed (e.g. Canopy, Anaconda, MacPorts),
that will allow easy and robust package management and avoid messing up with the 
system default python. I recommend using Anaconda over Canopy or MacPorts. 

The VIP package depends on existing packages from the Python ecosystem, e.g.:

numpy
scipy
matplotlib
pandas
astropy
scikit-learn
scikit-image
photutils
image_registration

OpenCV and it's python bindings are used for basic image processing operations. 
It's usually a difficult library to install (can't be processed from the setup.py 
requirements) but fortunately with just one command we'll have it ready.


INSTALLATION
============
Install OpenCV before running setup.py and make sure you have a C compiler 
installed in your system, like g++. With conda (anaconda) just type:
$ conda install opencv

A setup.py file that uses Setuptools Python package is included in the root 
folder. It takes care of installing the dependencies (see REQUIREMENTS) for you.

If you want to install VIP in your system run setup.py:
$ python setup.py install

The code is in continuous development and will be changing often. It's prefered 
to 'install' with the develop flag:
$ python setup.py develop

In any case wait a couple of minutes until all the requirements are satisfied.


FAQ
===
Why do I get and error about importing cv2 package when importing VIP?
R/ cv2 is the name of opencv bindings for python. This libraries are needed for
some fast image manipulations. Please open you Canopy package manager, search
and install opencv.

Why in linux do I get a matplotlib related error when importing VIP? 
[Error] Matplotlib backend_wx and backend_wxagg require wxPython >=2.8
R/ This is due to the interaction between linux/Canopy and python. Nothing to 
do with the VIP pipeline. You may need to try a different backend for 
matplotlib. Find your matplotlibrc configuration file and change the backend  
from WXAgg to Qt4Agg. More info here:
http://matplotlib.org/faq/usage_faq.html#what-is-a-backend







