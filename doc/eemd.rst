EEMD
====

Info
----

Ensemble empirical mode decomposition (EEMD) creates an ensemble of worker each
of which performs an :doc:`EMD </emd>` on a copy of the input signal with added noise.
When all workers finish their work a mean over all workers is considered as
the true result.

.. note::
    Given the nature of EEMD, each time you decompose a signal you will obtain a different set of components.
    That's the expected consequence of adding noise which is going to be random.
    To make the decomposition reproducible, one needs to set a seed for the random number generator used in EEMD.
    This is done using :func:`PyEMD.EEMD.noise_seed` method on the instance.

Class
-----

.. autoclass:: PyEMD.EEMD
    :members:
