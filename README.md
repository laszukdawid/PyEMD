[![codecov](https://codecov.io/gh/laszukdawid/PyEMD/branch/master/graph/badge.svg)](https://codecov.io/gh/laszukdawid/PyEMD)
[![Build Status](https://app.travis-ci.com/laszukdawid/PyEMD.svg?branch=master)](https://app.travis-ci.com/laszukdawid/PyEMD)
[![DocStatus](https://readthedocs.org/projects/pyemd/badge/?version=latest)](https://pyemd.readthedocs.io/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f56b6fc3f855476dbaebd3c02ae88f3e)](https://www.codacy.com/gh/laszukdawid/PyEMD/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=laszukdawid/PyEMD&amp;utm_campaign=Badge_Grade)
[![DOI](https://zenodo.org/badge/65324353.svg)](https://zenodo.org/badge/latestdoi/65324353)

# PyEMD

## Links

- Online documentation: <https://pyemd.readthedocs.org>
- Issue tracker: <https://github.com/laszukdawid/pyemd/issues>
- Source code repository: <https://github.com/laszukdawid/pyemd>

## Introduction

This is yet another Python implementation of Empirical Mode
Decomposition (EMD). The package contains many EMD variations and
intends to deliver more in time.

### EMD variations

-  Ensemble EMD (EEMD),
-  "Complete Ensemble EMD" (CEEMDAN)
-  different settings and configurations of vanilla EMD.
-  Image decomposition (EMD2D & BEMD) (experimental, no support)

*PyEMD* allows to use different splines for envelopes, stopping criteria
and extrema interpolation.

### Available splines

-  Natural cubic (**default**)
-  Pointwise cubic
-  Akima
-  Linear

### Available stopping criteria

-  Cauchy convergence (**default**)
-  Fixed number of iterations
-  Number of consecutive proto-imfs

### Extrema detection

-  Discrete extrema (**default**)
-  Parabolic interpolation

## Installation

### PyPi (recommended)

The quickest way to install package is through `pip`.

> \$ pip install EMD-signal

### From source

In case you only want to *use* EMD and its variation, the best way to install PyEMD is through `pip`.
However, if you are want to modify the code anyhow you might want to download the code and build package yourself.
The source is publicaly available and hosted on [GitHub](https://github.com/laszukdawid/PyEMD).
To download the code you can either go to the source code page and click `Code -> Download ZIP`, or use **git** command line

> \$ git clone <https://github.com/laszukdawid/PyEMD>

Installing package from source is done using command line:

> \$ python setup.py install

**Note**, however, that this will install it in your current environment. If you are working on many projects, or sharing reources with others, we suggest using [virtual environments](https://docs.python.org/3/library/venv.html).

## Example

More detailed examples are included in the
[documentation](https://pyemd.readthedocs.io/en/latest/examples.html) or
in the
[PyEMD/examples](https://github.com/laszukdawid/PyEMD/tree/master/example).

### EMD

In most cases default settings are enough. Simply import `EMD` and pass
your signal to instance or to `emd()` method.

```python
from PyEMD import EMD
import numpy as np

s = np.random.random(100)
emd = EMD()
IMFs = emd(s)
```

The Figure below was produced with input:
$S(t) = cos(22 \pi t^2) + 6t^2$

![simpleExample](https://github.com/laszukdawid/PyEMD/raw/master/example/simple_example.png?raw=true)

### EEMD

Simplest case of using Ensemble EMD (EEMD) is by importing `EEMD` and
passing your signal to the instance or `eemd()` method.

**Windows**: Please don't skip the `if __name__ == "__main__"` section. 

```python
from PyEMD import EEMD
import numpy as np

if __name__ == "__main__":
    s = np.random.random(100)
    eemd = EEMD()
    eIMFs = eemd(s)
```

### CEEMDAN

As with previous methods, there is also simple way to use `CEEMDAN`.

**Windows**: Please don't skip the `if __name__ == "__main__"` section. 

```python
from PyEMD import CEEMDAN
import numpy as np

if __name__ == "__main__":
    s = np.random.random(100)
    ceemdan = CEEMDAN()
    cIMFs = ceemdan(s)
```

### Visualisation

The package contain a simple visualisation helper that can help, e.g., with time series and instantaneous frequencies.

```python
import numpy as np
from PyEMD import EMD, Visualisation

t = np.arange(0, 3, 0.01)
S = np.sin(13*t + 0.2*t**1.4) - np.cos(3*t)

# Extract imfs and residue
# In case of EMD
emd = EMD()
emd.emd(S)
imfs, res = emd.get_imfs_and_residue()

# In general:
#components = EEMD()(S)
#imfs, res = components[:-1], components[-1]

vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
vis.plot_instant_freq(t, imfs=imfs)
vis.show()
```

### EMD2D/BEMD

*Unfortunately, this is Experimental and we can't guarantee that the output is meaningful.*
The simplest use is to pass image as monochromatic numpy 2D array. Sample as
with the other modules one can use the default setting of an instance or, more explicitly,
use the `emd2d()` method.

```python
from PyEMD.EMD2d import EMD2D  #, BEMD
import numpy as np

x, y = np.arange(128), np.arange(128).reshape((-1,1))
img = np.sin(0.1*x)*np.cos(0.2*y)
emd2d = EMD2D()  # BEMD() also works
IMFs_2D = emd2d(img)
```

## F.A.Q

### Why is EEMD/CEEMDAN so slow?
Unfortunately, that's their nature. They execute EMD multiple times every time with slightly modified version. Added noise can cause a creation of many extrema which will decrease performance of the natural cubic spline. For some tweaks on how to deal with that please see [Speedup tricks](https://pyemd.readthedocs.io/en/latest/speedup.html) in the documentation.

## Contact

Feel free to contact me with any questions, requests or simply to say *hi*.
It's always nice to know that I've helped someone or made their work easier. 
Contributing to the project is also acceptable and warmly welcomed.

### Citation

If you found this package useful and would like to cite it in your work
please use the following structure:

```latex
@misc{pyemd,
  author = {Laszuk, Dawid},
  title = {Python implementation of Empirical Mode Decomposition algorithm},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/laszukdawid/PyEMD}},
  doi = {10.5281/zenodo.5459184}
}
```
