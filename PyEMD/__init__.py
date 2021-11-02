import logging

__version__ = "1.1.1"
logger = logging.getLogger("pyemd")

from PyEMD.EMD import EMD
from PyEMD.EEMD import EEMD
from PyEMD.CEEMDAN import CEEMDAN
from PyEMD.visualisation import Visualisation
