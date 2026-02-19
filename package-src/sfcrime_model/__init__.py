from importlib.metadata import version

__all__ = ["__version__"]

try:
    __version__ = version("sfcrime-model")
except Exception:
    __version__ = "0.0.0"
