#!/usr/bin/env python
 
import os
import re
from setuptools import setup
try: # pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # pip <= 9.0.3
    from pip.req import parse_requirements
from setuptools.command.install import install


# Hackishly override of the install method
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

with open(os.path.join(PACKAGE_PATH, 'readme.rst')) as readme_file:
    README = readme_file.read()

with open(os.path.join(PACKAGE_PATH, 'vip_hci/__init__.py')) as version_file:
    version_file = version_file.read()
    VERSION = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                        version_file, re.M)
    VERSION = VERSION.group(1)


PACKAGES = ['vip_hci',
            'vip_hci.andromeda',
            'vip_hci.conf',
            'vip_hci.exlib',
            'vip_hci.fits',
            'vip_hci.frdiff',
            'vip_hci.leastsq',
            'vip_hci.llsg',
            'vip_hci.medsub',
            'vip_hci.metrics',
            'vip_hci.negfc',
            'vip_hci.nmf',
            'vip_hci.pca',
            'vip_hci.preproc',
            'vip_hci.stats',
            'vip_hci.var']

setup(
    name='vip_hci',
    version=VERSION,
    description='Package for astronomical high-contrast image processing.',
    long_description=README,
    license='MIT',
    author='Carlos Alberto Gomez Gonzalez',
    author_email='carlosgg33@gmail.com',
    url='https://github.com/vortex-exoplanet/VIP',
    cmdclass={'install': InstallReqs},
    packages=PACKAGES,
    install_requires=requirements,
    zip_safe=False,
    classifiers=['Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: POSIX :: Linux',
                 'Natural Language :: English',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering :: Astronomy'
                 ] 
)