EEMD
====

Info
----

Ensemble empirical mode decomposition (EEMD) creates an ensemble of worker each
of which performs an :doc:`EMD </emd>` on a copy of the input signal with added noise.
When all workers finish their work a mean over all workers is considered as
the true result.

.. note::
    **Parallel execution is enabled by default.** EEMD automatically uses all available
    CPU cores for faster computation. See :doc:`speedup </speedup>` for details on
    controlling parallelization.

.. note::
    Given the nature of EEMD, each time you decompose a signal you will obtain a different set of components.
    That's the expected consequence of adding noise which is going to be random.
    To make the decomposition reproducible, one needs to set a seed for the random number generator used in EEMD
    **and** set ``parallel=False``. This is done using :func:`PyEMD.EEMD.noise_seed` method on the instance::

        eemd = EEMD(parallel=False)
        eemd.noise_seed(12345)
        imfs = eemd(signal)

Class
-----

.. autoclass:: PyEMD.EEMD
    :members:
