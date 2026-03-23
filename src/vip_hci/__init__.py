import importlib as _importlib

_submodules = [
    "config",
    "fits",
    "fm",
    "greedy",
    "invprob",
    "metrics",
    "objects",
    "preproc",
    "psfsub",
    "stats",
    "var",
    "vip_ds9",
]


def __getattr__(name: str):
    if name in _submodules:
        return _importlib.import_module(f".{name}", __name__)
    if name == "__version__":
        from importlib.metadata import version
        return version("vip_hci")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return _submodules + ["__version__"]

