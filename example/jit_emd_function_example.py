import copy
import numpy as np

from PyEMD.experimental.jitemd import emd, default_emd_config
from PyEMD.experimental.jitemd import get_timeline

rng = np.random.RandomState(4132)
s = rng.random(500)
t = get_timeline(len(s), s.dtype)

config = copy.copy(default_emd_config)
config["FIXE"] = 5
imfs = emd(s, t, spline_kind="cubic", config=config)

print(imfs)
print(imfs.shape)
