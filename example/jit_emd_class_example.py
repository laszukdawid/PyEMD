import numpy as np

from PyEMD.experimental.jitemd import JitEMD, get_timeline

rng = np.random.RandomState(4132)
s = rng.random(500)
t = get_timeline(len(s), s.dtype)

emd = JitEMD(spline_kind="akima")

imfs = emd(s, t)

print(imfs)
print(imfs.shape)
