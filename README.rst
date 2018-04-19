|codecov| |BuildStatus| |DocStatus| |Codacy|


*****
PyEMD
*****

Links
*****
- HTML documentation: https://pyemd.readthedocs.org
- Issue tracker: https://github.com/laszukdawid/pyemd/issues
- Source code repository: https://github.com/laszukdawid/pyemd

Introduction
************

This is yet another Python implementation of Empirical Mode
Decomposition (EMD). The package contains many EMD variations and intends to
deliver more in time.

EMD variations:
    - Ensemble EMD (EEMD),
    - Image decomposotion (EMD2D),
    - "Complete Ensembl EMD" (CEEMDAN)
    - different settings and configurations of vanilla EMD.

*PyEMD* allows to use different splines for envelopes, stopping criteria
and extrema interpolation.

Available splines:
    - Natural cubic [default] 
    - Pointwise cubic 
    - Akima 
    - Linear

Available stopping criteria: 
    - Cauchy convergence [default] 
    - Fixed number of iterations 
    - Number of consecutive proto-imfs

Extrema detection: 
    - Discrete extrema [default] 
    - Parabolic interpolation

Installation
************

Recommended
===========

Simply download this directory either directly from GitHub, or using command line:

    $ git clone https://github.com/laszukdawid/PyEMD

Then go into the downloaded project and run from command line:

    $ python setup.py install


PyPi
====
Packaged obtained from PyPi is/will be slightly behind this project, so some features might not be the same. However, it seems to be the easiest/nicest way of installing any Python packages, so why not this one?

    $ pip install EMD-signal


Example
*******

More detailed examples are included in the documentation_ or in the `PyEMD/examples`_.

EMD
===

In most cases default settings are enough. Simply
import ``EMD`` and pass your signal to instance or to ``emd()`` method.

.. code:: python

    from PyEMD import EMD
    import numpy as np

    s = np.random.random(100)
    emd = EMD()
    IMFs = emd(s)

The Figure below was produced with input:
:math:`S(t) = cos(22 \pi t^2) + 6t^2` 

|simpleExample|

EEMD
====

Simplest case of using Esnembld EMD (EEMD) is by importing ``EEMD`` and passing
your signal to the instance or ``eemd()`` method.

.. code:: python

    from PyEMD import EEMD
    import numpy as np

    s = np.random.random(100)
    eemd = EEMD()
    eIMFs = eemd(s)

CEEMDAN
=======

As with previous methods, there is also simple way to use ``CEEMDAN``.

.. code:: python

    from PyEMD import CEEMDAN
    import numpy as np

    s = np.random.random(100)
    ceemdan = CEEMDAN()
    cIMFs = ceemdan(s)

EMD2D
=====

Simplest case is to pass image as monochromatic numpy 2D array.
As with other modules one can use default setting of instance or
more expliclity use ``emd2d()`` method.

.. code:: python

    from PyEMD import EMD2D
    import numpy as np

    x, y = np.arange(128), np.arange(128).reshape((-1,1))
    img = np.sin(0.1*x)*np.cos(0.2*y)
    emd2d = EMD2D()
    IMFs_2D = emd2d(img)

Contact
*******

Feel free to contact me with any questions, requests or simply saying
*hi*. It's always nice to know that I might have contributed to saving
someone's time or that I might improve my skills/projects.

Contact me either through gmail (laszukdawid @ gmail) or search me
favourite web search.

Citation
========

If you found this package useful and would like to cite it in your work
please use following structure:

Dawid Laszuk (2017-), **Python implementation of Empirical Mode Decomposition algorithm**. http://www.laszukdawid.com/codes.


.. |codecov| image:: https://codecov.io/gh/laszukdawid/PyEMD/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/laszukdawid/PyEMD
.. |BuildStatus| image:: https://travis-ci.org/laszukdawid/PyEMD.png?branch=master
   :target: https://travis-ci.org/laszukdawid/PyEMD
.. |DocStatus| image:: https://readthedocs.org/projects/pyemd/badge/?version=latest
   :target: https://pyemd.readthedocs.io/
   
.. |Codacy| image:: https://api.codacy.com/project/badge/Grade/5385d5ddc8e84908bd4e38f325443a21
    :alt: Codacy Badge
    :target: https://www.codacy.com/app/laszukdawid/PyEMD?utm_source=github.com&utm_medium=referral&utm_content=laszukdawid/PyEMD&utm_campaign=badger
.. |simpleExample| image:: https://github.com/laszukdawid/PyEMD/raw/master/example/simple_example.png?raw=true
.. _documentation: https://pyemd.readthedocs.io/en/latest/examples.html
.. _`PyEMD/examples`: https://github.com/laszukdawid/PyEMD/tree/master/example
