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
    To make the decomposition reproducible, set a seed using
    :func:`PyEMD.EEMD.noise_seed` on the instance. Noise is drawn in the parent
    process and trial results are collected in order, so the same seed produces
    the same ensemble with serial or parallel execution::

        eemd = EEMD()
        eemd.noise_seed(12345)
        imfs = eemd(signal)

Class
-----

.. autoclass:: PyEMD.EEMD
    :members:
