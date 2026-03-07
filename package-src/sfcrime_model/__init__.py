from pathlib import Path

__all__ = ["__version__"]

with open(Path(__file__).resolve().parent / "VERSION") as _f:
    __version__: str = _f.read().strip()
