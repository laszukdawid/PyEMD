CEEMDAN
=======

Info
----

Complete ensemble EMD with adaptive noise (CEEMDAN) performs an EEMD with
the difference that the information about the noise is shared among all workers.


.. note::
    Given the nature of CEEMDAN, each time you decompose a signal you will obtain a different set of components.
    That's the expected consequence of adding noise which is going to be random and different.
    To make the decomposition reproducible, one needs to set a seed for the random number generator used in CEEMDAN.
    This is done using :func:`PyEMD.CEEMDAN.noise_seed` method on the instance.

Class
-----

.. autoclass:: PyEMD.CEEMDAN
    :members:
