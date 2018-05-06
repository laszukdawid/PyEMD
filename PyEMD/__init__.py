import logging

from PyEMD.EMD import EMD
from PyEMD.EEMD import EEMD
from PyEMD.EMD2d import EMD2D
from PyEMD.BEMD import BEMD
from PyEMD.CEEMDAN import CEEMDAN

logger = logging.getLogger('pyemd')
logger.addHandler(logging.NullHandler())
if len(logger.handlers)==1:
    logging.basicConfig(level=logging.INFO)

