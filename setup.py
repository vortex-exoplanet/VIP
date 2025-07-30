#!/usr/bin/env python
import os

from setuptools import setup
try:
    # pip >=20
    from pip._internal.network.session import PipSession
    from pip._internal.req import parse_requirements
except ImportError:
    try:
        # 10.0.0 <= pip <= 19.3.1
        from pip._internal.download import PipSession
        from pip._internal.req import parse_requirements
    except ImportError:
        # pip <= 9.0.3
        from pip.download import PipSession
        from pip.req import parse_requirements
from setuptools.command.install import install
from setuptools.command.develop import develop


# Hackishly override of the install method
class InstallReqs(install):
    def run(self):
        print(" ********************** ")
        print(" *** Installing VIP *** ")
        print(" ********************** ")
        os.system('pip install -r requirements.txt')
        install.run(self)


class InstallDevReqs(develop):
    def run(self):
        print(" **************************** ")
        print(" *** Installing VIP (dev) *** ")
        print(" **************************** ")
        os.system('pip install -r requirements-dev.txt')
        develop.run(self)


def resource(*args):
    return os.path.join(os.path.abspath(os.path.join(__file__, os.pardir)),
                        *args)


# parse_requirements() returns generator of pip.req.InstallRequirement objects
reqs = parse_requirements(resource('requirements.txt'), session=PipSession)
try:
    requirements = [str(ir.requirement) for ir in reqs]
except:
    requirements = [str(ir.req) for ir in reqs]

reqs_dev = parse_requirements(resource('requirements-dev.txt'),
                              session=PipSession)
try:
    requirements_dev = [str(ir.requirement) for ir in reqs_dev]
except:
    requirements_dev = [str(ir.req) for ir in reqs_dev]


setup(
    cmdclass={'install': InstallReqs,
              'develop': InstallDevReqs},
    install_requires=requirements,
    extras_require={"dev": requirements_dev},
)
