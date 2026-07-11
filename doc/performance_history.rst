Performance improvement history
===============================

This page records performance improvements that were evaluated for PyEMD,
including changes that were not merged because they did not improve measured
runtime. The goal is to keep optimization work reproducible and to avoid
repeating attempts that already showed neutral or negative results.

How to run focused benchmarks
-----------------------------

The focused EMD accumulation benchmark can be run with::

    python perf_test/perf_test_comprehensive.py --test accumulation

It executes ``test_emd_accumulation`` with these default parameters:

* ``signal_length=5000``
* ``runs=50``
* ``warmup=5``
* ``complexity="complex"``

The generated signal combines several sinusoidal components, random noise, and
a linear trend. The benchmark records the number of returned components so that
results can be interpreted against the amount of IMF accumulation performed.

The focused BEMD extrema value extraction benchmark can be run with::

    python perf_test/perf_test_comprehensive.py --test bemd_values

It executes ``test_bemd_extrema_value_extraction`` with these default
parameters:

* ``image_size=256``
* ``extrema_count=4000``
* ``runs=200``
* ``warmup=10``

This benchmark isolates value extraction in ``BEMD.extract_max_min_spline`` by
stubbing spline construction. If the optional BEMD dependencies are not
installed, the benchmark is skipped.

Saved benchmark runs can be compared with::

    python perf_test/compare_results.py <baseline-results> <comparison-results>

Attempt: replace EMD vstack accumulation with list accumulation
---------------------------------------------------------------

Date evaluated
    2026-07-11

Area
    ``PyEMD.EMD.emd`` component accumulation and residue reconstruction.

Motivation
    ``EMD.emd`` appended each extracted IMF with ``np.vstack``. Repeatedly
    growing NumPy arrays can be expensive because every append copies existing
    data. A common optimization is to collect arrays in a Python list and stack
    once at the end.

Attempted implementation
    The implementation was changed to collect IMFs in a list. A second variant
    also maintained the residue incrementally so the end-condition check could
    use the current residue instead of recomputing ``S - np.sum(IMF, axis=0)``.

Correctness result
    After handling the empty-IMF trend case, the full test suite passed::

        python -m PyEMD.tests.test_all
        Ran 116 tests
        OK (skipped=21)

Performance result
    The focused benchmark did not improve. Comparing the original baseline with
    the best attempted implementation showed a statistically significant
    regression::

        EMD_accumulation ({'signal_length': 5000, 'complexity': 'complex'}):
        0.0679s -> 0.0729s (+7.3% slower, p=0.000)

Decision
    The implementation change was not kept. The benchmark was kept so future
    EMD accumulation changes can be measured before they are merged.

Interpretation
    For the evaluated signal, ``np.vstack`` was not the dominant cost. The run
    returned 8 components, so array growth happened only a small number of
    times. Runtime is more likely dominated by repeated extrema detection,
    envelope spline interpolation, and sifting iterations.

Future notes
    List accumulation may still be worth re-evaluating for workloads that
    produce many more IMFs or much larger arrays, but it should be judged by a
    benchmark result rather than assumed to be faster.

Attempt: vectorize BEMD extrema value extraction
------------------------------------------------

Date evaluated
    2026-07-11

Area
    ``PyEMD.BEMD.extract_max_min_spline`` value extraction for minima and
    maxima positions.

Motivation
    The method extracted image values with Python list comprehensions over
    coordinate pairs::

        np.array([image[x, y] for x, y in zip(*min_peaks_pos)])

    NumPy supports advanced indexing with the coordinate tuple directly, which
    avoids Python-level iteration.

Implementation
    The list comprehensions were replaced with direct indexed reads::

        min_val = image[min_peaks_pos]
        max_val = image[max_peaks_pos]

Correctness result
    The full test suite passed::

        python -m PyEMD.tests.test_all
        Ran 116 tests
        OK (skipped=21)

Performance result
    Comparing the original baseline with the vectorized implementation showed a
    statistically significant improvement::

        BEMD_extrema_value_extraction ({'image_size': 256, 'extrema_count': 4000}):
        0.0035s -> 0.0015s (-55.8% faster, p=0.000)

    The comparison reported high variance, so exact timings should not be
    treated as absolute. The effect size was still large and positive.

Decision
    The implementation change was kept.

Interpretation
    This is a narrow optimization in an experimental 2D path. It improves the
    value extraction portion of envelope construction, but full BEMD runtime may
    still be dominated by extrema detection and radial-basis interpolation.
