EEMD
====

Info
----
Ensemble empirical mode decomposition (EEMD) creates an ensemble of worker each
of which performs an EMD on a copy of the input signal with added noise.
When all workers finish their work a mean over all workers is considered as
the true result.

Class
-----

.. autoclass:: PyEMD.EEMD
    :members:
    :special-members:
