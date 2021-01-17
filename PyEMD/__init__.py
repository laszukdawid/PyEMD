import logging

__version__ = "0.2.12"
logger = logging.getLogger('pyemd')

from PyEMD.EMD import EMD
from PyEMD.EEMD import EEMD
from PyEMD.CEEMDAN import CEEMDAN
from PyEMD.visualisation import Visualisation
