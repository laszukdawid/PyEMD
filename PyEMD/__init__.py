import logging

logger = logging.getLogger('pyemd')
logger.addHandler(logging.NullHandler())
if len(logger.handlers)==1:
    logging.basicConfig(level=logging.INFO)

from PyEMD.EMD import EMD
from PyEMD.EEMD import EEMD
from PyEMD.CEEMDAN import CEEMDAN
from PyEMD.visualisation import Visualisation

try:
    from PyEMD.EMD2d import EMD2D
    from PyEMD.BEMD import BEMD
except ImportError:
    logger.debug("EMD2D and BEMD are not supported in the basic version. They are dependent on packages in the `requriements-extra`.")
