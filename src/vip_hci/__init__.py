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


def __getattr__(name: str):
    if name == '__version__':
        from importlib.metadata import version
        return version('vip_hci')
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
