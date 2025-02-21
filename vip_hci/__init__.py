from . import preproc
from . import config
from . import fits
from . import invprob
from . import psfsub
from . import fm
from . import metrics
from . import stats
from . import var
from . import objects
from .vip_ds9 import *

try:
    from ._version import __version__
except ImportError:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version(__name__)
    except PackageNotFoundError:
        __version__ = "0.0.0"  # Default version
