[![Coverage Status](https://coveralls.io/repos/github/laszukdawid/PyEMD/badge.svg?branch=master)](https://coveralls.io/github/laszukdawid/PyEMD?branch=master)
[![codecov](https://codecov.io/gh/laszukdawid/PyEMD/branch/master/graph/badge.svg)](https://codecov.io/gh/laszukdawid/PyEMD)
[![Build Status](https://travis-ci.org/laszukdawid/PyEMD.png?branch=master)](https://travis-ci.org/laszukdawid/PyEMD)

# PyEMD
<span style="background-color: #D8D8D8">The project is ongoing. This is very limited part of my private collection, but before I upload everything I want to make sure it works as it should. If there is something you wish to have, do email me as there is high chance that I have already done it, but it just sits around and waits until I'll have more time. Don't hesitate contacting me for anything.</span>

This is yet another Python implementation of Empirical Mode Decomposition (EMD).
The package contains many EMD variations, like Ensemble EMD (EEMD), and different settings.

_PyEMD_ allows to use different splines for envelopes, stopping criteria and extrema interpolation.

Available splines:
* Natural cubic [default]
* Pointwise cubic
* Akima
* Linear

Available stopping criteria:
* Cauchy convergence [default]
* Fixed number of iterations
* Number of consecutive proto-imfs

Extrema detection:
* Discrete extrema [default]
* Parabolic interpolation


# Example
Probably in most cases default settings are enough.
In such case simply import `EMD` and pass your signal to `emd()` method.
```python
from PyEMD import EMD

s = np.random.random(100)
IMFs = EMD().emd(s)
```

The Figure below was produced with input:
$S(t) = cos(22 \pi t^2) + 6t^2$
![simple_example](PyEMD/example/simple_example.png?raw=true "Simple example of EMD results")

# Contact
Feel free to contact me with any questions, requests or simply saying _hi_.
It's always nice to know that I might have contributed to saving someone's time or that I might improve my skills/projects.

Contact me either through gmail ({my_username}@gmail) or search me favourite web search.
