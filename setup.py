#!/usr/bin/env python
 
import os
from setuptools import setup, find_packages
from pip.req import parse_requirements
from setuptools.command.install import install

class InstallReqs(install):
    def run(self):
        print(" ********************** ")
        print(" *** Installing VIP *** ")
        print(" ********************** ")
        os.system('pip install -r requirements')
        install.run(self)

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements(os.path.join(PACKAGE_PATH, 'requirements'),
                                  session=False)
requirements = [str(ir.req) for ir in install_reqs]

with open(os.path.join(PACKAGE_PATH, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()

with open(os.path.join(PACKAGE_PATH, 'readme.rst')) as readme_file:
    readme = readme_file.read()
 
setup(
    name='vip',
    version=__version__,
    description='Package for astronomical high-contrast image processing.',
    long_description=readme,
    author='Carlos Gomez Gonzalez',
    author_email='cgomez@ulg.ac.be',
    url='https://github.com/vortex-exoplanet/VIP',
    cmdclass={'install': InstallReqs},
    packages=find_packages(),
    install_requires=requirements,
    zip_safe=False,
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: POSIX :: Linux',
                 'Natural Language :: English',
                 'Programming Language :: Python :: 2.7',
                 'Topic :: Scientific/Engineering :: Astronomy'
                 ] 
)