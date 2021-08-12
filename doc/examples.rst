Examples
********

Some examples can be found in PyEMD/example directory.

EMD
===

Quick start
-----------
In most cases default settings are enough. Simply
import :py:class:`EMD` and pass your signal to `emd` method.

.. code:: python

    from PyEMD import EMD

    s = np.random.random(100)
    emd = EMD()
    IMFs = emd.emd(s)

Something more
``````````````
Here is a complete script on how to create and plot results.

.. code:: python

    from PyEMD import EMD
    import numpy  as np
    import pylab as plt

    # Define signal
    t = np.linspace(0, 1, 200)
    s = np.cos(11*2*np.pi*t*t) + 6*t*t

    # Execute EMD on signal
    IMF = EMD().emd(s,t)
    N = IMF.shape[0]+1

    # Plot results
    plt.subplot(N,1,1)
    plt.plot(t, s, 'r')
    plt.title("Input signal: $S(t)=cos(22\pi t^2) + 6t^2$")
    plt.xlabel("Time [s]")

    for n, imf in enumerate(IMF):
        plt.subplot(N,1,n+2)
        plt.plot(t, imf, 'g')
        plt.title("IMF "+str(n+1))
        plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.savefig('simple_example')
    plt.show()


The Figure below was produced with input:

:math:`S(t) = cos(22 \pi t^2) + 6t^2` 

|simpleExample|

EEMD
====

Simplest case of using Ensemble EMD (EEMD) is by importing `EEMD` and passing your signal to `eemd` method.

.. code:: python

    from PyEMD import EEMD
    import numpy as np
    import pylab as plt

    # Define signal
    t = np.linspace(0, 1, 200)

    sin = lambda x,p: np.sin(2*np.pi*x*t+p)
    S = 3*sin(18,0.2)*(t-0.2)**2
    S += 5*sin(11,2.7)
    S += 3*sin(14,1.6)
    S += 1*np.sin(4*2*np.pi*(t-0.8)**2)
    S += t**2.1 -t

    # Assign EEMD to `eemd` variable 
    eemd = EEMD()

    # Say we want detect extrema using parabolic method
    emd = eemd.EMD
    emd.extrema_detection="parabol"

    # Execute EEMD on S
    eIMFs = eemd.eemd(S, t)
    nIMFs = eIMFs.shape[0]

    # Plot results
    plt.figure(figsize=(12,9))
    plt.subplot(nIMFs+1, 1, 1)
    plt.plot(t, S, 'r')

    for n in range(nIMFs):
        plt.subplot(nIMFs+1, 1, n+2)
        plt.plot(t, eIMFs[n], 'g')
        plt.ylabel("eIMF %i" %(n+1))
        plt.locator_params(axis='y', nbins=5)

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig('eemd_example', dpi=120)
    plt.show()

|eemdExample|


.. |simpleExample| image:: https://github.com/laszukdawid/PyEMD/raw/master/example/simple_example.png
    :align: middle
    :alt: Oh, the quality. Please click on the image for better resolution.
    :target: https://github.com/laszukdawid/PyEMD/raw/master/example/simple_example.png
 
.. |eemdExample| image:: https://github.com/laszukdawid/PyEMD/raw/master/example/eemd_example.png?raw=true
    :width: 720px
    :height: 540px
