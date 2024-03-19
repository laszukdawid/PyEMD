import logging

__version__ = "1.6.0"  # noqa
logger = logging.getLogger("pyemd")  # noqa

from PyEMD.CEEMDAN import CEEMDAN  # noqa
from PyEMD.EEMD import EEMD  # noqa
from PyEMD.EMD import EMD  # noqa
from PyEMD.visualisation import Visualisation  # noqa
