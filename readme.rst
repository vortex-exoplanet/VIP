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

VIP is a package/pipeline for angular and spectral differential high-contrast 
imaging written in Python 2.7.

VIP is being developed within the VORTEX team @ University of Liege (Belgium).
It's in alpha version meaning that the code will change drastically before the
first release version. If you want to report a bug contact the main developer
at cgomez [at] ulg.ac.be or through github.


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


DOCUMENTATION 
==============
Docstrings (docuementation of every function/class) are included in the code.
Sphinx can be used to generate the documentation from the docstrings in html or
pdf into docs folder (To-Do).


USAGE
======
For interactive mode execute Ipython shell and import the package:
>>> import vip

>>> vip.<TAB> 
shows all the subpackages available,

>>> vip.<subpackage>.<function or class name>? 
for displaying the docstring (notice the ? question mark). This is a fast and
convenient way to obtain help.


TO-DO
=====
- Register in Pypi for even easier installation.
- Better interface, OOP structure.
- Proper html/pdf documentation, include more Jupyter(Ipython) tutorial notebooks.
- Usage of pytest testing tool for better code consistency. 
- Migrate to python 3.


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







