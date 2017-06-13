Intro
=====

General
-------

**PyEMD** is a Python implementation of `Empirical Mode Decomposition (EMD) <https://en.wikipedia.org/wiki/Hilbert%E2%80%93Huang_transform#Techniques>`_ and its variations.
One of the most popular expansion is `Ensemble Empirical Mode Decomposition (EEMD) <http://www.worldscientific.com/doi/abs/10.1142/S1793536909000047>`_, which utilises an ensemble of noise-assisted executions.

As a result of *EMD* one will obtain a set of components that possess oscillatory features. In case of plain *EMD* algorithm, these are called Intrinsic Mode Functions (IMFs) as they are expected to have a single `mode <https://en.wikipedia.org/wiki/Normal_mode>`_. In contrary, *EEMD* will unlikely produce pure oscillations as the effects of injected noise can propagate throughout the decomposition. 

Installation
------------

Recommended
```````````

In order to get the newest version it is recommended to download source code from git repository. **Don't worry, installation is simple.**
Simply download this directory either directly from GitHub, or using command line: ::

    $ git clone https://github.com/laszukdawid/PyEMD

Then go into the downloaded project and run from command line: ::

    $ python setup.py install


PyPi (simplest)
```````````````

Packaged obtained from PyPi is/will be slightly behind this project, so some features might not be the same. However, it seems to be the easiest/nicest way of installing any Python packages, so why not this one? ::

    $ pip install EMD-signal


