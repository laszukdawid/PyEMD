Usage
=============

Typical Usage
-------------

Majority, if not all, methods follow the same usage pattern:

* Import method
* Initiate method
* Apply method on data

On vanilla EMD this is as

.. code-block:: python

    from PyEMD import EMD
    emd = EMD()
    imfs = emd(s)

Parameters
----------

The decomposition can be changed by adjusting parameters related to either sifting or stopping conditions.

Sifting
```````
The sifting depends on the used method so these parameters ought to be looked within the methods.
However, the typical parameters relate to spline method or the number of mirroring points.


Stopping conditions
```````````````````
All methods have the same two conditions, `FIXE` and `FIXE_H`, for stopping which relate to the number of sifting iterations.
Setting parameter `FIXE` to any positive value will fix the number of iterations for each IMF to be exactly `FIXE`.

Example:

.. code-block:: python

    emd = EMD()
    emd.FIXE = 10
    imfs = emd(s)

Parameter `FIXE_H` relates to the number of iterations when the proto-IMF signal fulfils IMF conditions, i.e. number of extrema and zero-crossings differ at most by one and the mean is close to zero. This means that there will be at least `FIXE_H` iteration per IMF.

Example:

.. code-block:: python

    emd = EMD()
    emd.FIXE_H = 5
    imfs = emd(s)

When both `FIXE` and `FIXE_H` are 0 then other conditions are checked. These can be checking for convergence between consecutive iterations or whether the amplitude of output is below acceptable range.
