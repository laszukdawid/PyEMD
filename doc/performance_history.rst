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

The focused 3-point cubic spline benchmark can be run with::

    python perf_test/perf_test_comprehensive.py --test cubic3pts

It executes ``test_cubic_spline_3pts`` with these default parameters:

* ``runs=2000``
* ``warmup=100``
* ``timeline_length=512``

This benchmark isolates the small fallback used when cubic spline envelopes have
only three extrema points.

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

Attempt: use solve instead of explicit inverse for 3-point cubic spline
-----------------------------------------------------------------------

Date evaluated
    2026-07-11

Area
    ``PyEMD.splines.cubic_spline_3pts`` coefficient solve.

Motivation
    The fallback spline for exactly three extrema points computed coefficients
    with an explicit matrix inverse followed by a dot product::

        np.linalg.inv(M).dot(v)

    Solving the linear system directly avoids constructing the inverse and is
    the standard NumPy approach for this operation.

Implementation
    The inverse/dot expression was replaced with::

        np.linalg.solve(M, v)

Correctness result
    A direct old-vs-new equivalence check over randomized inputs preserved the
    output timeline exactly and matched spline values within ``1e-12`` absolute
    and relative tolerance. The full test suite also passed::

        python -m PyEMD.tests.test_all
        Ran 116 tests
        OK (skipped=21)

Performance result
    Comparing the original baseline with the direct-solve implementation showed
    a statistically significant improvement in the focused benchmark::

        cubic_spline_3pts ({'timeline_length': 512, 'points': 3}):
        0.0001s -> 0.0001s (-7.3% faster, p=0.000)

    The benchmark reported high variance and the absolute runtime is very
    small, so the percentage should be treated as directional rather than a
    precise end-to-end speedup estimate.

Decision
    The implementation change was kept because it is both faster in the focused
    benchmark and numerically preferable to explicitly inverting the matrix.

Interpretation
    This is a small local improvement. It only affects envelope construction
    cases with exactly three extrema points, so full EMD runtime impact depends
    on how often a decomposition reaches that fallback path.
