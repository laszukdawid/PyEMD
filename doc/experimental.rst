Experimental
============

Also known as **not supported**.

Methods discussed and provided here have no guarantee to work or provide any meaningful results.
These are somehow abandoned projects and unfortunately mid-way through. They aren't completely
discarded simply because of hope that maybe someday someone will come and help fix them.
We all know that the best motivation to do something is to be annoyed by the current state.
Seriously though, mode decomposition in 2D and multi-dim is an interesting topic. Please?

JIT EMD
-------
.. note::

    To use JitEMD you need to install PyEMD wiht ``jit`` option.
    If you're using ``pip`` then use this command

    .. code:: shell

        pip install EMD-signal[jit]

Uses Numba (https://numba.pydata.org/) as a Just-in-Time (JIT)
compiler for Python, mostly Numpy. Just-in-time compilation means that
the code is compiled (machine code) during execution, and thus shows
benefit when there's plenty of repeated code or same code used a lot.

This EMD implementation is experimental as it only provides value
when there's significant amount of computation required, e.g. when
analyzing HUGE time series with a lot of internal complexity,
or reuses the instance/method many times, e.g. in a script,
iPython REPL or jupyter notebook.

Additional reason for this being experimental is that the author (me)
isn't well veristile in Numba optimization. There's definitely a lot
that can be improved. It's being added as maybe it'll be helpful or
an inspiration for others to learn something and contribute to the PyEMD.

Comparison
**********

There's an `example <https://github.com/laszukdawid/PyEMD/tree/master/example/emd_comparison.py>`_ which compares the pefromance between JIT and classic EMD.

    ❯ python example/emd_comparison.py 

    Comparing EEMD execution on a larger signal with classic and JIT EMDs.
    Signal is random (uniform) noise of length: 2000.
    The test is done by executing EEMD with either classic or JIT EMD 20 times
    and taking the average. Such setup favouries JitEMD which is compiled once
    and then reused 19 times. Compiltion is quite costly.

    Classic EEMD on 2000 length random signal:   5.7 s per EEMD run

    JitEMD EEMD on 2000 length random signal:   4.2 per EEMD run

Usage
*****

There are two ways of interacting with JIT EMD; either as a function,
or as a class compatible with the rest of PyEMD ecosystem. Please take
a look at `code examples <https://github.com/laszukdawid/PyEMD/tree/master/example>`_ for a quick start.

Class JitEMD
************

When using class ``experimental.JitEMD`` it'll be compatible with other PyEMD classes, for example with EEMD.
That's why it'll accept the same inputs and will provide the same outputs.
The only difference is in configuring the class. It now has to be a copy of ``default_emd_config``
with updated values.

.. code:: python

    from PyEMD.experimental.jitemd import default_emd_config, JitEMD, get_timeline

    rng = np.random.RandomState(4132)
    s = rng.random(500)
    t = get_timeline(len(s), s.dtype)

    config = default_emd_config
    config["FIXE"] = 4
    emd = JitEMD(config=config, spline_kind="akima")
    imfs = emd(s, t)


Function JIT emd
****************

When using ``emd`` function directly, you only get the *imfs* as the result and only once.
There's also a differnce where the ``config`` is passed; it's directly to the method.
Other than that, it should be the same. The class ``JitEMD`` uses this function and provides
some abstraction on to of it for, hopefully, easier use, but there might be benefits to
access the function directly.

.. code:: python

    from PyEMD.experimental.jitemd import default_emd_config, emd, get_timeline

    s = rng.random(500)
    t = get_timeline(len(s), s.dtype)

    config = default_emd_config
    config["FIXE"] = 4
    imfs = emd(s, t, spline_kind="cubic", config=config)

BEMD
----

.. warning::

    Important This is an experimental module. Please use it with care as no
    guarantee can be given for obtaining reasonable results, or that they will be
    computed index the most computation optimal way.

Info
****

**BEMD** performed on bidimensional data such as images.
This procedure uses morphological operators to detect regional maxima
which are then used to span surface envelope with a radial basis function.

Class
*****

.. autoclass:: PyEMD.BEMD.BEMD
    :members:
    :special-members:
    
EMD2D
-----

.. warning::

    Important This is an experimental module. Please use it with care as no
    guarantee can be given for obtaining reasonable results, or that they will be
    computed index the most computation optimal way.

Info
****

**EMD** performed on images. This version uses for envelopes 2D splines,
which are span on extrema defined through maximum filter.

Class
*****

.. autoclass:: PyEMD.EMD2d.EMD2D
    :members:
    :special-members:
