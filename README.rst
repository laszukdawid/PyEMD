|codecov| |BuildStatus| |DocStatus|


*****
PyEMD
*****

The project is ongoing. This is very limited part of my private
collection, but before I upload everything I want to make sure it works
as it should. If there is something you wish to have, do email me as
there is high chance that I have already done it, but it just sits
around and waits until I'll have more time. Don't hesitate contacting me
for anything.

This is yet another Python implementation of Empirical Mode
Decomposition (EMD). The package contains many EMD variations, like
Ensemble EMD (EEMD), and different settings.

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

EMD
===

In most cases default settings are enough. Simply
import ``EMD`` and pass your signal to ``emd()`` method.

.. code:: python

    from PyEMD import EMD

    s = np.random.random(100)
    emd = EMD()
    IMFs = emd.emd(s)

The Figure below was produced with input:
:math:`S(t) = cos(22 \pi t^2) + 6t^2` 

|simpleExample|

EEMD
====

Simplest case of using Esnembld EMD (EEMD) is by importing ``EEMD`` and passing your signal to ``eemd()`` method.

.. code:: python

    from PyEMD import EEMD

    s = np.random.random(100)
    eemd = EEMD()
    eIMFs = eemd.eemd(s)

Contact
*******

Feel free to contact me with any questions, requests or simply saying
*hi*. It's always nice to know that I might have contributed to saving
someone's time or that I might improve my skills/projects.

Contact me either through gmail ({my\_username}@gmail) or search me
favourite web search.


.. |codecov| image:: https://codecov.io/gh/laszukdawid/PyEMD/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/laszukdawid/PyEMD
.. |BuildStatus| image:: https://travis-ci.org/laszukdawid/PyEMD.png?branch=master
   :target: https://travis-ci.org/laszukdawid/PyEMD
.. |DocStatus| image:: https://readthedocs.org/projects/pyemd/badge/?version=latest
   :target https://pyemd.readthedocs.io/
.. |simpleExample| image:: https://github.com/laszukdawid/PyEMD/raw/master/PyEMD/example/simple_example.png?raw=true
