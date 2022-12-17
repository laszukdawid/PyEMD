from typing import Dict

import numpy as np

from PyEMD.fast_emd import emd, get_timeline, EmdConfig

s = np.random.random(100)
t = get_timeline(len(s), s.dtype)

config = EmdConfig()

imfs = emd(s, t, spline_kind="cubic")

print(imfs)
print(imfs.shape)
