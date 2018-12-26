[![codecov](https://codecov.io/gh/laszukdawid/PyEMD/branch/master/graph/badge.svg)](https://codecov.io/gh/laszukdawid/PyEMD)
[![BuildStatus](https://travis-ci.org/laszukdawid/PyEMD.png?branch=master)](https://travis-ci.org/laszukdawid/PyEMD)
[![DocStatus](https://readthedocs.org/projects/pyemd/badge/?version=latest)](https://pyemd.readthedocs.io/)
[![Codacy](https://api.codacy.com/project/badge/Grade/5385d5ddc8e84908bd4e38f325443a21)](https://www.codacy.com/app/laszukdawid/PyEMD?utm_source=github.com&utm_medium=referral&utm_content=laszukdawid/PyEMD&utm_campaign=badger)

# PyEMD

## Links

-   HTML documentation: <https://pyemd.readthedocs.org>
-   Issue tracker: <https://github.com/laszukdawid/pyemd/issues>
-   Source code repository: <https://github.com/laszukdawid/pyemd>

## Introduction

This is yet another Python implementation of Empirical Mode
Decomposition (EMD). The package contains many EMD variations and
intends to deliver more in time.

### EMD variations:
* Ensemble EMD (EEMD),
* "Complete Ensemble EMD" (CEEMDAN)
* different settings and configurations of vanilla EMD.
* Image decomposition (EMD2D & BEMD) (experimental)

*PyEMD* allows to use different splines for envelopes, stopping criteria
and extrema interpolation.

### Available splines:
* Natural cubic [default]
* Pointwise cubic
* Akima
* Linear

### Available stopping criteria:
* Cauchy convergence [default]
* Fixed number of iterations
* Number of consecutive proto-imfs

### Extrema detection:
* Discrete extrema [default]
* Parabolic interpolation

## Installation

### Recommended

Simply download this directory either directly from GitHub, or using
command line:

> \$ git clone <https://github.com/laszukdawid/PyEMD>

Then go into the downloaded project and run from command line:

> \$ python setup.py install

### PyPi

Packaged obtained from PyPi is/will be slightly behind this project, so
some features might not be the same. However, it seems to be the
easiest/nicest way of installing any Python packages, so why not this
one?

> \$ pip install EMD-signal

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

```python
from PyEMD import EEMD
import numpy as np

s = np.random.random(100)
eemd = EEMD()
eIMFs = eemd(s)
```

### CEEMDAN

As with previous methods, there is also simple way to use `CEEMDAN`.

```python
from PyEMD import CEEMDAN
import numpy as np

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
from PyEMD import EMD2D  #, BEMD
import numpy as np

x, y = np.arange(128), np.arange(128).reshape((-1,1))
img = np.sin(0.1*x)*np.cos(0.2*y)
emd2d = EMD2D()  # BEMD() also works
IMFs_2D = emd2d(img)
```

## Contact

Feel free to contact me with any questions, requests or simply to say
*hi*. It's always nice to know that I one's work have eased others and saved
someone's time. Contributing to the project is also acceptable.

Contact me either through gmail (laszukdawid @ gmail) or search me through your
favourite web search.

### Citation

If you found this package useful and would like to cite it in your work
please use following structure:

Dawid Laszuk (2017-), **Python implementation of Empirical Mode
Decomposition algorithm**. <http://www.laszukdawid.com/codes>.
