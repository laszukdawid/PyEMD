import numpy as np

from PyEMD import EEMD
from PyEMD.experimental.jitemd import JitEMD, get_timeline

s = np.random.random(100)
t = get_timeline(len(s), s.dtype)

emd = JitEMD()
eemd = EEMD(ext_EMD=emd)
imfs = eemd(s, t)

print(imfs)
print(imfs.shape)
