"""
Comprehensive Performance Test Suite for PyEMD

This module tests various performance aspects of the EMD library:
1. EMD scaling with signal length
2. EMD with different spline methods
3. EMD with different extrema detection methods
4. EEMD parallel scaling
5. CEEMDAN performance characteristics

Run with: .venv/bin/python perf_test/perf_test_comprehensive.py

Results are saved to perf_test/results/<timestamp>/ directory.
"""

import json
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from PyEMD import CEEMDAN, EEMD, EMD

# Results directory
RESULTS_BASE_DIR = Path(__file__).parent / "results"

# Default random seed for reproducibility
DEFAULT_SEED = 42


def reset_random_state(seed: int = DEFAULT_SEED):
    """Reset all random number generators for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class PerfResult:
    """Container for performance test results with statistics."""

    name: str
    params: Dict
    mean: float  # Mean time in seconds
    std: float  # Standard deviation
    min: float  # Minimum time
    max: float  # Maximum time
    runs: int  # Number of timed runs (excluding warmup)
    trimmed_mean: float  # 10% trimmed mean for outlier robustness
    extra: Optional[Dict] = None

    def __str__(self) -> str:
        extra_str = f", {self.extra}" if self.extra else ""
        return f"{self.name}: {self.mean:.4f}s ± {self.std:.4f}s (trimmed={self.trimmed_mean:.4f}s, min={self.min:.4f}, max={self.max:.4f}, n={self.runs}) ({self.params}{extra_str})"

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "params": self.params,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "runs": self.runs,
            "trimmed_mean": self.trimmed_mean,
            "extra": self.extra,
        }


def get_system_info() -> Dict:
    """Collect system information for reproducibility."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

    # Try to get git commit hash
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent.parent,
            )
            .decode()
            .strip()
        )
        info["git_commit"] = git_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_commit"] = "unknown"

    # Get PyEMD version
    try:
        from PyEMD import __version__

        info["pyemd_version"] = __version__
    except ImportError:
        info["pyemd_version"] = "unknown"

    # Get numpy version
    info["numpy_version"] = np.__version__

    return info


def create_results_dir(prefix: str = "") -> Path:
    """Create a timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        dirname = f"{timestamp}_{prefix}"
    else:
        dirname = timestamp

    results_dir = RESULTS_BASE_DIR / dirname
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_results(results: List[PerfResult], results_dir: Path, system_info: Dict):
    """Save results to JSON and human-readable text files."""
    # Save as JSON
    json_data = {"system_info": system_info, "results": [r.to_dict() for r in results]}
    with open(results_dir / "results.json", "w") as f:
        json.dump(json_data, f, indent=2)

    # Save human-readable summary
    with open(results_dir / "summary.txt", "w") as f:
        f.write("PyEMD Performance Test Results\n")
        f.write("=" * 60 + "\n\n")

        f.write("System Information:\n")
        f.write("-" * 40 + "\n")
        for key, value in system_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        # Group results by test name
        grouped = {}
        for r in results:
            if r.name not in grouped:
                grouped[r.name] = []
            grouped[r.name].append(r)

        for test_name, test_results in grouped.items():
            f.write(f"\n{test_name}:\n")
            f.write("-" * 40 + "\n")
            for r in test_results:
                f.write(f"  {r}\n")

    print(f"\nResults saved to: {results_dir}")


def trimmed_mean(times: List[float], trim_percent: float = 0.1) -> float:
    """Calculate trimmed mean by removing extreme values.

    Args:
        times: List of timing values
        trim_percent: Fraction to trim from each end (default 10% from each side)

    Returns:
        Trimmed mean value
    """
    if len(times) < 3:
        return float(np.mean(times))

    sorted_times = sorted(times)
    n = len(sorted_times)
    trim_count = max(1, int(n * trim_percent))

    # Ensure we keep at least one value
    if 2 * trim_count >= n:
        trim_count = max(0, (n - 1) // 2)

    trimmed = sorted_times[trim_count : n - trim_count] if trim_count > 0 else sorted_times
    return float(np.mean(trimmed))


@dataclass
class BenchmarkStats:
    """Statistics from a benchmark run."""

    times: List[float]
    mean: float
    std: float
    min: float
    max: float
    trimmed_mean: float  # 10% trimmed mean for outlier robustness

    @classmethod
    def from_times(cls, times: List[float], trim_percent: float = 0.1) -> "BenchmarkStats":
        return cls(
            times=times,
            mean=float(np.mean(times)),
            std=float(np.std(times)),
            min=float(np.min(times)),
            max=float(np.max(times)),
            trimmed_mean=trimmed_mean(times, trim_percent),
        )


def benchmark(
    func: Callable,
    *args,
    runs: int = 5,
    warmup: int = 1,
    seed: int = DEFAULT_SEED,
    instance_with_seed: object = None,
    **kwargs,
) -> BenchmarkStats:
    """Benchmark a function with warmup runs and statistics.

    Args:
        func: Function to benchmark
        *args: Arguments to pass to func
        runs: Number of timed runs (after warmup)
        warmup: Number of warmup runs (not timed)
        seed: Random seed to reset before each run for reproducibility
        instance_with_seed: Optional object with noise_seed() method (EEMD/CEEMDAN)
        **kwargs: Keyword arguments to pass to func

    Returns:
        BenchmarkStats with timing statistics
    """
    # Warmup runs (not timed)
    for i in range(warmup):
        reset_random_state(seed + i)  # Different seed for warmup, but deterministic
        if instance_with_seed is not None and hasattr(instance_with_seed, "noise_seed"):
            instance_with_seed.noise_seed(seed + i)
        func(*args, **kwargs)

    # Timed runs - reset seed before EACH run for identical conditions
    times = []
    for i in range(runs):
        run_seed = seed + warmup + i
        reset_random_state(run_seed)  # Deterministic but different per run
        if instance_with_seed is not None and hasattr(instance_with_seed, "noise_seed"):
            instance_with_seed.noise_seed(run_seed)
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return BenchmarkStats.from_times(times)


def generate_test_signal(n: int, complexity: str = "medium") -> np.ndarray:
    """Generate test signals of varying complexity.

    Args:
        n: Signal length
        complexity: One of "simple", "medium", "complex"
    """
    t = np.linspace(0, 1, n)

    if complexity == "simple":
        # Single frequency
        return np.sin(2 * np.pi * 5 * t)
    elif complexity == "medium":
        # Two frequencies + trend
        return np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + t
    elif complexity == "complex":
        # Multiple frequencies + noise + trend
        return (
            np.sin(2 * np.pi * 5 * t)
            + 0.5 * np.sin(2 * np.pi * 20 * t)
            + 0.3 * np.sin(2 * np.pi * 50 * t)
            + 0.1 * np.random.randn(n)
            + 2 * t
        )
    else:
        raise ValueError(f"Unknown complexity: {complexity}")


# =============================================================================
# Test 1: EMD Scaling with Signal Length
# =============================================================================


def test_emd_scaling(
    signal_lengths: List[int] = None, runs: int = 200, warmup: int = 10
) -> List[PerfResult]:
    """Test how EMD performance scales with signal length."""
    if signal_lengths is None:
        signal_lengths = [500, 1000, 2000, 5000, 10000]

    results = []
    emd = EMD()

    for n in signal_lengths:
        signal = generate_test_signal(n, "medium")
        stats = benchmark(emd.emd, signal, runs=runs, warmup=warmup)

        # Get IMF count from last run
        imfs = emd.emd(signal)
        n_imfs = imfs.shape[0]

        results.append(
            PerfResult(
                name="EMD_scaling",
                params={"signal_length": n},
                mean=stats.mean,
                std=stats.std,
                min=stats.min,
                max=stats.max,
                runs=runs,
                trimmed_mean=stats.trimmed_mean,
                extra={"n_imfs": n_imfs},
            )
        )

    return results


# =============================================================================
# Test 2: Spline Method Comparison
# =============================================================================


def test_spline_methods(
    signal_length: int = 2000, runs: int = 50, warmup: int = 3
) -> List[PerfResult]:
    """Compare performance of different spline interpolation methods."""
    spline_kinds = ["cubic", "akima", "pchip", "cubic_hermite", "slinear"]
    signal = generate_test_signal(signal_length, "medium")

    results = []

    for spline_kind in spline_kinds:
        emd = EMD(spline_kind=spline_kind)
        try:
            stats = benchmark(emd.emd, signal, runs=runs, warmup=warmup)
            results.append(
                PerfResult(
                    name="spline_comparison",
                    params={"spline_kind": spline_kind, "signal_length": signal_length},
                    mean=stats.mean,
                    std=stats.std,
                    min=stats.min,
                    max=stats.max,
                    runs=runs,
                    trimmed_mean=stats.trimmed_mean,
                )
            )
        except Exception as e:
            print(f"  Spline '{spline_kind}' failed: {e}")

    return results


# =============================================================================
# Test 3: Extrema Detection Comparison
# =============================================================================


def test_extrema_detection(
    signal_length: int = 2000, runs: int = 5, warmup: int = 3
) -> List[PerfResult]:
    """Compare 'simple' vs 'parabol' extrema detection methods."""
    methods = ["simple", "parabol"]
    signal = generate_test_signal(signal_length, "medium")

    results = []

    for method in methods:
        emd = EMD(extrema_detection=method)
        stats = benchmark(emd.emd, signal, runs=runs, warmup=warmup)
        results.append(
            PerfResult(
                name="extrema_detection",
                params={"method": method, "signal_length": signal_length},
                mean=stats.mean,
                std=stats.std,
                min=stats.min,
                max=stats.max,
                runs=runs,
                trimmed_mean=stats.trimmed_mean,
            )
        )

    return results


# =============================================================================
# Test 4: EEMD Parallel Scaling
# =============================================================================


def test_eemd_parallel(
    signal_length: int = 1000,
    trial_counts: List[int] = None,
    process_counts: List[int] = None,
    runs: int = 30,
    warmup: int = 2,
) -> List[PerfResult]:
    """Test EEMD parallel scaling with different trial and process counts."""
    if trial_counts is None:
        trial_counts = [10, 25, 50]
    if process_counts is None:
        process_counts = [1, 2, 4]

    signal = generate_test_signal(signal_length, "medium")
    results = []

    # Sequential baseline
    for trials in trial_counts:
        eemd = EEMD(trials=trials, parallel=False)
        stats = benchmark(
            eemd.eemd, signal, runs=runs, warmup=warmup, instance_with_seed=eemd
        )
        results.append(
            PerfResult(
                name="EEMD_sequential",
                params={"trials": trials, "signal_length": signal_length},
                mean=stats.mean,
                std=stats.std,
                min=stats.min,
                max=stats.max,
                runs=runs,
                trimmed_mean=stats.trimmed_mean,
            )
        )

    # Parallel tests
    for trials in trial_counts:
        for processes in process_counts:
            if processes == 1:
                continue  # Already tested as sequential
            eemd = EEMD(trials=trials, parallel=True, processes=processes)
            stats = benchmark(
                eemd.eemd, signal, runs=runs, warmup=warmup, instance_with_seed=eemd
            )
            results.append(
                PerfResult(
                    name="EEMD_parallel",
                    params={
                        "trials": trials,
                        "processes": processes,
                        "signal_length": signal_length,
                    },
                    mean=stats.mean,
                    std=stats.std,
                    min=stats.min,
                    max=stats.max,
                    runs=runs,
                    trimmed_mean=stats.trimmed_mean,
                )
            )

    return results


# =============================================================================
# Test 5: CEEMDAN Performance
# =============================================================================


def test_ceemdan(
    signal_length: int = 500,
    trial_counts: List[int] = None,
    runs: int = 3,
    warmup: int = 2,
) -> List[PerfResult]:
    """Test CEEMDAN performance with different trial counts.

    Note: CEEMDAN has quadratic complexity in trials due to nested loops.
    """
    if trial_counts is None:
        trial_counts = [5, 10, 25]

    signal = generate_test_signal(signal_length, "medium")
    results = []

    for trials in trial_counts:
        ceemdan = CEEMDAN(trials=trials)
        stats = benchmark(
            ceemdan.ceemdan,
            signal,
            runs=runs,
            warmup=warmup,
            instance_with_seed=ceemdan,
        )

        # Get IMF count
        ceemdan.noise_seed(DEFAULT_SEED)
        imfs = ceemdan.ceemdan(signal)
        n_imfs = imfs.shape[0]

        results.append(
            PerfResult(
                name="CEEMDAN",
                params={"trials": trials, "signal_length": signal_length},
                mean=stats.mean,
                std=stats.std,
                min=stats.min,
                max=stats.max,
                runs=runs,
                trimmed_mean=stats.trimmed_mean,
                extra={"n_imfs": n_imfs},
            )
        )

    return results


# =============================================================================
# Test 6: Signal Complexity Impact
# =============================================================================


def test_signal_complexity(
    signal_length: int = 2000, runs: int = 5, warmup: int = 3
) -> List[PerfResult]:
    """Test how signal complexity affects EMD performance."""
    complexities = ["simple", "medium", "complex"]
    results = []
    emd = EMD()

    for complexity in complexities:
        signal = generate_test_signal(signal_length, complexity)
        stats = benchmark(emd.emd, signal, runs=runs, warmup=warmup)

        # Get stats from last run
        imfs = emd.emd(signal)
        n_imfs = imfs.shape[0]

        results.append(
            PerfResult(
                name="signal_complexity",
                params={"complexity": complexity, "signal_length": signal_length},
                mean=stats.mean,
                std=stats.std,
                min=stats.min,
                max=stats.max,
                runs=runs,
                trimmed_mean=stats.trimmed_mean,
                extra={"n_imfs": n_imfs},
            )
        )

    return results


# =============================================================================
# Test 7: Sifting Parameters Impact
# =============================================================================


def test_sifting_params(
    signal_length: int = 2000, runs: int = 5, warmup: int = 3
) -> List[PerfResult]:
    """Test impact of sifting parameters on performance."""
    results = []
    signal = generate_test_signal(signal_length, "medium")

    # Test different MAX_ITERATION values
    max_iterations = [100, 500, 1000, 2000]
    for max_iter in max_iterations:
        emd = EMD(MAX_ITERATION=max_iter)
        stats = benchmark(emd.emd, signal, runs=runs, warmup=warmup)
        results.append(
            PerfResult(
                name="max_iteration",
                params={"MAX_ITERATION": max_iter, "signal_length": signal_length},
                mean=stats.mean,
                std=stats.std,
                min=stats.min,
                max=stats.max,
                runs=runs,
                trimmed_mean=stats.trimmed_mean,
            )
        )

    # Test FIXE_H (fixed number of sifting iterations)
    fixe_h_values = [0, 5, 10, 20]
    for fixe_h in fixe_h_values:
        emd = EMD(FIXE_H=fixe_h)
        stats = benchmark(emd.emd, signal, runs=runs, warmup=warmup)
        results.append(
            PerfResult(
                name="fixe_h",
                params={"FIXE_H": fixe_h, "signal_length": signal_length},
                mean=stats.mean,
                std=stats.std,
                min=stats.min,
                max=stats.max,
                runs=runs,
                trimmed_mean=stats.trimmed_mean,
            )
        )

    return results


# =============================================================================
# Main Runner
# =============================================================================


def print_results(results: List[PerfResult], title: str):
    """Pretty print test results."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)
    for r in results:
        print(f"  {r}")
    print()


def run_all_tests(quick: bool = False, save: bool = True) -> List[PerfResult]:
    """Run all performance tests.

    Args:
        quick: If True, run with smaller parameters for faster feedback
        save: If True, save results to timestamped directory

    Returns:
        List of all performance results
    """
    # Reset random state at start for reproducible signal generation
    reset_random_state()

    print("PyEMD Comprehensive Performance Test Suite")
    print("=" * 60)

    # Collect system info
    system_info = get_system_info()
    print(f"Timestamp: {system_info['timestamp']}")
    print(f"Git commit: {system_info['git_commit'][:8]}...")
    print(f"PyEMD version: {system_info['pyemd_version']}")

    if quick:
        print("\nRunning in QUICK mode (smaller parameters)")
        signal_lengths = [500, 1000, 2000]
        eemd_trials = [5, 10]
        eemd_processes = [1, 2]
        ceemdan_trials = [3, 5]
        runs = 10
        warmup = 3
        eemd_runs = 5
        eemd_warmup = 2
        prefix = "quick"
    else:
        print("\nRunning FULL test suite")
        signal_lengths = [500, 1000, 2000, 5000, 10000]
        eemd_trials = [10, 25, 50]
        eemd_processes = [1, 2, 4]
        ceemdan_trials = [5, 10, 25]
        runs = 15
        warmup = 5
        eemd_runs = 7
        eemd_warmup = 3
        prefix = "full"

    all_results = []

    # Test 1: EMD Scaling
    print("\n[1/7] Testing EMD scaling with signal length...")
    results = test_emd_scaling(signal_lengths, runs=runs * 2, warmup=warmup)
    print_results(results, "EMD Scaling Test")
    all_results.extend(results)

    # Test 2: Spline Methods
    print("[2/7] Testing spline interpolation methods...")
    results = test_spline_methods(signal_length=2000, runs=runs * 2, warmup=warmup)
    print_results(results, "Spline Method Comparison")
    all_results.extend(results)

    # Test 3: Extrema Detection
    print("[3/7] Testing extrema detection methods...")
    results = test_extrema_detection(signal_length=2000, runs=runs * 2, warmup=warmup)
    print_results(results, "Extrema Detection Comparison")
    all_results.extend(results)

    # Test 4: EEMD Parallel
    print("[4/7] Testing EEMD parallel scaling...")
    results = test_eemd_parallel(
        signal_length=1000,
        trial_counts=eemd_trials,
        process_counts=eemd_processes,
        runs=eemd_runs,
        warmup=eemd_warmup,
    )
    print_results(results, "EEMD Parallel Scaling")
    all_results.extend(results)

    # Test 5: CEEMDAN
    print("[5/7] Testing CEEMDAN performance...")
    results = test_ceemdan(
        signal_length=500,
        trial_counts=ceemdan_trials,
        runs=eemd_runs,
        warmup=eemd_warmup,
    )
    print_results(results, "CEEMDAN Performance")
    all_results.extend(results)

    # Test 6: Signal Complexity
    print("[6/7] Testing signal complexity impact...")
    results = test_signal_complexity(signal_length=2000, runs=runs, warmup=warmup)
    print_results(results, "Signal Complexity Impact")
    all_results.extend(results)

    # Test 7: Sifting Parameters
    print("[7/7] Testing sifting parameters...")
    results = test_sifting_params(signal_length=2000, runs=runs, warmup=warmup)
    print_results(results, "Sifting Parameters Impact")
    all_results.extend(results)

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"Total tests run: {len(all_results)}")

    # Save results
    if save:
        results_dir = create_results_dir(prefix)
        save_results(all_results, results_dir, system_info)

    return all_results


def run_single_test(test_name: str, save: bool = True) -> List[PerfResult]:
    """Run a single test by name.

    Args:
        test_name: One of 'scaling', 'splines', 'extrema', 'eemd', 'ceemdan',
                   'complexity', 'sifting'
        save: If True, save results to timestamped directory

    Returns:
        List of performance results
    """
    # Reset random state for reproducibility
    reset_random_state()

    system_info = get_system_info()

    test_map = {
        "scaling": (test_emd_scaling, "EMD Scaling Test"),
        "splines": (test_spline_methods, "Spline Method Comparison"),
        "extrema": (test_extrema_detection, "Extrema Detection Comparison"),
        "eemd": (test_eemd_parallel, "EEMD Parallel Scaling"),
        "ceemdan": (test_ceemdan, "CEEMDAN Performance"),
        "complexity": (test_signal_complexity, "Signal Complexity Impact"),
        "sifting": (test_sifting_params, "Sifting Parameters Impact"),
    }

    if test_name not in test_map:
        raise ValueError(
            f"Unknown test: {test_name}. Choose from: {list(test_map.keys())}"
        )

    func, title = test_map[test_name]
    results = func()
    print_results(results, title)

    if save:
        results_dir = create_results_dir(test_name)
        save_results(results, results_dir, system_info)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PyEMD Performance Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python perf_test_comprehensive.py                    # Full test suite
  python perf_test_comprehensive.py --quick            # Quick test suite
  python perf_test_comprehensive.py --test scaling     # Single test
  python perf_test_comprehensive.py --no-save          # Don't save results
  python perf_test_comprehensive.py --profile --quick  # Profile quick suite
  python perf_test_comprehensive.py --profile --test scaling  # Profile single test

Results are saved to: perf_test/results/<timestamp>_<prefix>/
        """,
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick tests with smaller parameters"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=[
            "scaling",
            "splines",
            "extrema",
            "eemd",
            "ceemdan",
            "complexity",
            "sifting",
            "all",
        ],
        default="all",
        help="Which test to run (default: all)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save results to disk"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run test suite with cProfile profiling"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show detailed timing statistics for EMD"
    )

    args = parser.parse_args()
    save = not args.no_save

    if args.stats:
        print("Detailed EMD timing statistics")
        print("=" * 70)
        reset_random_state()
        signal = generate_test_signal(2000, "medium")
        emd = EMD()

        # Run many iterations and collect all times
        runs = 100
        times = []
        for i in range(runs):
            reset_random_state(DEFAULT_SEED + i)
            start = time.perf_counter()
            emd.emd(signal)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times_arr = np.array(times)
        mean = np.mean(times_arr)
        std = np.std(times_arr)
        cv = (std / mean) * 100

        print(f"\nSignal length: 2000, Runs: {runs}")
        print(f"Mean:   {mean*1000:.3f} ms")
        print(f"Std:    {std*1000:.3f} ms")
        print(f"CV:     {cv:.1f}%")
        print(f"Min:    {np.min(times_arr)*1000:.3f} ms")
        print(f"Max:    {np.max(times_arr)*1000:.3f} ms")
        print(f"Median: {np.median(times_arr)*1000:.3f} ms")

        # Percentiles
        print("\nPercentiles:")
        for p in [5, 25, 50, 75, 95]:
            val = np.percentile(times_arr, p)
            print(f"  {p:3d}%: {val*1000:.3f} ms")

        # Show distribution of times
        print("\nTime distribution (histogram):")
        hist, edges = np.histogram(times_arr * 1000, bins=10)
        for i, count in enumerate(hist):
            bar = "#" * int(count * 40 / max(hist))
            print(f"  {edges[i]:5.2f}-{edges[i+1]:5.2f} ms: {bar} ({count})")

    elif args.profile:
        import cProfile
        import pstats

        if args.test == "all":
            test_desc = "quick test suite" if args.quick else "full test suite"
        else:
            test_desc = f"'{args.test}' test"
        print(f"Running profiled {test_desc}...")
        print("=" * 70)

        profiler = cProfile.Profile()
        profiler.enable()

        if args.test == "all":
            run_all_tests(quick=args.quick, save=save)
        else:
            run_single_test(args.test, save=save)

        profiler.disable()

        print("\n" + "=" * 70)
        print(" PROFILING RESULTS")
        print("=" * 70)

        print("\nTop 30 functions by cumulative time:")
        print("-" * 70)
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)

        print("\nTop 30 functions by total time:")
        print("-" * 70)
        stats.strip_dirs().sort_stats("tottime").print_stats(30)
    elif args.test == "all":
        run_all_tests(quick=args.quick, save=save)
    else:
        run_single_test(args.test, save=save)
