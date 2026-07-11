import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("EMD-signal")
except PackageNotFoundError:
    __version__ = "0+unknown"

logger = logging.getLogger("pyemd")

from PyEMD.CEEMDAN import CEEMDAN  # noqa
from PyEMD.EEMD import EEMD  # noqa
from PyEMD.EMD import EMD  # noqa
from PyEMD.visualisation import Visualisation  # noqa
