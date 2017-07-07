import logging

from PyEMD.EMD import EMD
from PyEMD.EEMD import EEMD
from PyEMD.EMD2d import EMD2D

logger = logging.getLogger('pyemd')
logger.addHandler(logging.NullHandler())
if len(logger.handlers)==1:
    logging.basicConfig(level=logging.INFO)

