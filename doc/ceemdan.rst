CEEMDAN
=======

Info
----

Complete ensemble EMD with adaptive noise (CEEMDAN) performs an EEMD with
the difference that the information about the noise is shared among all workers.

.. note::
    **Parallel execution is enabled by default.** CEEMDAN automatically uses all available
    CPU cores for faster computation. See :doc:`speedup </speedup>` for details on
    controlling parallelization.

.. note::
    Given the nature of CEEMDAN, each time you decompose a signal you will obtain a different set of components.
    That's the expected consequence of adding noise which is going to be random and different.
    To make the decomposition reproducible, one needs to set a seed for the random number generator used in CEEMDAN
    **and** set ``parallel=False``. This is done using :func:`PyEMD.CEEMDAN.noise_seed` method on the instance::

        ceemdan = CEEMDAN(parallel=False)
        ceemdan.noise_seed(12345)
        imfs = ceemdan(signal)

Class
-----

.. autoclass:: PyEMD.CEEMDAN
    :members:
