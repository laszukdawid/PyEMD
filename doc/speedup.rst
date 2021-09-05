Speedup tricks
==============

EMD is inherently slow with little chances on improving its performance. This is mainly due to it being a serial method. That's both on within IMF stage, i.e. iterative sifting, or between IMFs, i.e. the next IMF depends on the previous. On top of that, the common configuration of the EMD uses the natural cubic spline to span envelopes, which in turn additionally decreases performance since it depends on all extrema in the signal.

Since the EMD is the basis for other methods like EEMD and CEEMDAN these will also suffer from the same problem. What's more, these two methods perform the EMD many (hundreds) times which significantly increases any imperfections. It is expected that when it'll take more than a minute to perform an EEMD/CEEMDAN with default settings on a 10k+ samples long signal with a "medium complexity". There are, however, a couple of tweaks one can do to do make the computation finish sooner.

Sections below describe a tweaks one can do to improve performance of the EMD. In short, these changes are:

- `Change data type`_ (downscale)
- `Change spline method`_ to piecewise
- `Decrease number of trials`_
- `Limit numer of output IMFs`_


Change data type
----------------

Many programming frameworks by default casts numerical values to the largest data type it has. In case of Python's Numpy that's going to be numpy.float64. It's unlikely that one needs such resolution when using EMD [*]_. A suggestion is to downcast your data, e.g. to float16. The PyEMD should handle the same data type without upcasting but it can be additionally enforce a specific data type.  To enable data type enforcement one needs to pass the DTYPE, i.e. ::

    from PyEMD import EMD

    emd = EMD(DTYPE=np.float16)

Change spline method
--------------------

EMD was presented with the natural cubic spline method to span envelopes and that's the default option in the PyEMD. It's great for signals with not many extrema but its not suggested for longer/more complex signals. The suggestion is to change the spline method to some piecewise splines like 'Akima' or 'piecewise cubic'.

Example: ::

    from PyEMD import EEMD

    eemd = EEMD(spline_kind='akima')

Decrease number of trials
----------------------------

This relates more to EEMD and CEEMDAN since they perform an EMD a multiple times with slightly modified signal. It's difficult to choose a correct number of iterations. This definitely relates to the signal in question. The more iterations the more certain that the solution is convergent but there is likely a point beyond which more evaluations change little. On the other side, the quicker we can get output the quicker we can use it.

In the PyEMD, the number of iterations is referred to by `trials` and it's an explicit parameter to EEMD and CEEMDAN. The default value was selected arbitrarily and it's most likely wrong. An example on updating it: ::

    from PyEMD import CEEMDAN

    ceemdan = CEEMAN(trials=20)

Limit numer of output IMFs
--------------------------

Each method, by default, will perform decomposition until all components are returned. However, many use cases only require the first component. One can limit the number of returned components by setting up an implicit variable `max_imf` to the desired value.

Example: ::

    from PyEMD import EEMD

    eemd = EEMD(max_imfs=2)


.. [*] I, the PyEMD's author, will go even a bit further. If one needs such large resolution then the EMD is not suitable for them. The EMD is not robust. Hundreds of iterations make any small difference to be emphasised and potentially leading to a significant change in final decomposition. This is the reason for creating EEMD and CEEMDAN which add small perturbation in a hope that the ensemble provides a robust solution.
