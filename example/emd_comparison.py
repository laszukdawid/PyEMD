"""
This example shows very high numbers for JIT because on import the JitEMD is compiled
making it very efficient. It takes a good 20+ seconds to compile which is hidden.

For this reason, JitEMD is worthy when iterating on live notebook or performing
a lot of the same computation within a single execution. For a single script
with a single EMD execution, it's still much more performant to use normal EMD.
"""
import time
import numpy as np

from PyEMD import EEMD, EMD
from PyEMD.experimental.jitemd import get_timeline
from PyEMD.experimental.jitemd import JitEMD

s = np.random.random(2000)
t = get_timeline(len(s), s.dtype)
n_repeat = 20

print(f"""Comparing EEMD execution on a larger signal with classic and JIT EMDs.
Signal is random (uniform) noise of length: {len(s)}. The test is done by executing
EEMD with either classic or JIT EMD {n_repeat} times and taking the average. Such
setup favouries JitEMD which is compiled once and then reused {n_repeat-1} times.
Compiltion is quite costly.""")

time_0 = time.time()
emd = EMD()
eemd = EEMD(ext_EMD=emd)
for _ in range(n_repeat):
    _ = eemd(s, t)
time_1 = time.time()
t_per_one = (time_1 - time_0) / n_repeat
print(f"Classic EEMD on {len(s)} length random signal: {t_per_one:5.2} s per EEMD run")

time_0 = time.time()
emd = JitEMD()
eemd = EEMD(ext_EMD=emd)
for _ in range(n_repeat):
    _ = eemd(s, t)
time_1 = time.time()
t_per_one = (time_1 - time_0) / n_repeat
print(f"JitEMD EEMD on {len(s)} length random signal: {t_per_one:5.2} per EEMD run")
