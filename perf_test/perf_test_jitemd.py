"""
Performance Test Suite for JIT-compiled EMD (jitemd.py)

This module tests various performance aspects of the JIT EMD implementation:
1. JIT compilation warmup overhead
2. JitEMD scaling with signal length
3. JitEMD vs standard EMD comparison
4. JitEMD with different spline methods
5. JitEMD with different extrema detection methods
6. Repeated execution benefits (JIT advantage)

Run with: .venv/bin/python perf_test/perf_test_jitemd.py

Results are saved to perf_test/results/<timestamp>_jitemd/ directory.
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

from PyEMD import EMD
from PyEMD.experimental.jitemd import JitEMD, default_emd_config, emd as jit_emd_func

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

    # Get numpy and numba versions
    info["numpy_version"] = np.__version__

    try:
        import numba

        info["numba_version"] = numba.__version__
    except ImportError:
        info["numba_version"] = "not installed"

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
        f.write("PyEMD JitEMD Performance Test Results\n")
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
    """Calculate trimmed mean by removing extreme values."""
    if len(times) < 3:
        return float(np.mean(times))

    sorted_times = sorted(times)
    n = len(sorted_times)
    trim_count = max(1, int(n * trim_percent))

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
    trimmed_mean: float

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
    **kwargs,
) -> BenchmarkStats:
    """Benchmark a function with warmup runs and statistics."""
    # Warmup runs (not timed)
    for i in range(warmup):
        reset_random_state(seed + i)
        func(*args, **kwargs)

    # Timed runs
    times = []
    for i in range(runs):
        run_seed = seed + warmup + i
        reset_random_state(run_seed)
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return BenchmarkStats.from_times(times)


def generate_test_signal(n: int, complexity: str = "medium", dtype=np.float64) -> np.ndarray:
    """Generate test signals of varying complexity."""
    t = np.linspace(0, 1, n, dtype=dtype)

    if complexity == "simple":
        return np.sin(2 * np.pi * 5 * t).astype(dtype)
    elif complexity == "medium":
        return (np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + t).astype(dtype)
    elif complexity == "complex":
        return (
            np.sin(2 * np.pi * 5 * t)
            + 0.5 * np.sin(2 * np.pi * 20 * t)
            + 0.3 * np.sin(2 * np.pi * 50 * t)
            + 0.1 * np.random.randn(n)
            + 2 * t
        ).astype(dtype)
    else:
        raise ValueError(f"Unknown complexity: {complexity}")


# =============================================================================
# Test 1: JIT Compilation Warmup Overhead
# =============================================================================


def test_jit_warmup(signal_length: int = 1000, runs: int = 10) -> List[PerfResult]:
    """Test JIT compilation warmup overhead.

    Measures the first call (includes compilation) vs subsequent calls.
    """
    results = []
    signal = generate_test_signal(signal_length, "medium")
    t = np.linspace(0, 1, signal_length, dtype=np.float64)

    # Force fresh compilation by creating new instance
    jit_emd = JitEMD()

    # First run (includes JIT compilation)
    reset_random_state()
    start = time.perf_counter()
    jit_emd.emd(signal, t)
    first_run_time = time.perf_counter() - start

    # Subsequent runs (JIT already compiled)
    subsequent_times = []
    for i in range(runs):
        reset_random_state(DEFAULT_SEED + i)
        start = time.perf_counter()
        jit_emd.emd(signal, t)
        elapsed = time.perf_counter() - start
        subsequent_times.append(elapsed)

    stats = BenchmarkStats.from_times(subsequent_times)

    results.append(
        PerfResult(
            name="JIT_warmup",
            params={"signal_length": signal_length, "phase": "first_run"},
            mean=first_run_time,
            std=0.0,
            min=first_run_time,
            max=first_run_time,
            runs=1,
            trimmed_mean=first_run_time,
            extra={"includes_compilation": True},
        )
    )

    results.append(
        PerfResult(
            name="JIT_warmup",
            params={"signal_length": signal_length, "phase": "subsequent"},
            mean=stats.mean,
            std=stats.std,
            min=stats.min,
            max=stats.max,
            runs=runs,
            trimmed_mean=stats.trimmed_mean,
            extra={"includes_compilation": False},
        )
    )

    # Calculate warmup overhead
    warmup_overhead = first_run_time / stats.mean if stats.mean > 0 else float("inf")
    results.append(
        PerfResult(
            name="JIT_warmup",
            params={"signal_length": signal_length, "phase": "overhead_ratio"},
            mean=warmup_overhead,
            std=0.0,
            min=warmup_overhead,
            max=warmup_overhead,
            runs=1,
            trimmed_mean=warmup_overhead,
            extra={"first_vs_subsequent": f"{warmup_overhead:.1f}x slower"},
        )
    )

    return results


# =============================================================================
# Test 2: JitEMD Scaling with Signal Length
# =============================================================================


def test_jitemd_scaling(signal_lengths: List[int] = None, runs: int = 20, warmup: int = 5) -> List[PerfResult]:
    """Test how JitEMD performance scales with signal length."""
    if signal_lengths is None:
        signal_lengths = [500, 1000, 2000, 5000, 10000]

    results = []
    jit_emd = JitEMD()

    # Pre-warm JIT compilation with a small signal
    warm_signal = generate_test_signal(100, "medium")
    warm_t = np.linspace(0, 1, 100, dtype=np.float64)
    jit_emd.emd(warm_signal, warm_t)

    for n in signal_lengths:
        signal = generate_test_signal(n, "medium")
        t = np.linspace(0, 1, n, dtype=np.float64)

        stats = benchmark(jit_emd.emd, signal, t, runs=runs, warmup=warmup)

        # Get IMF count from last run
        imfs = jit_emd.emd(signal, t)
        n_imfs = imfs.shape[0]

        results.append(
            PerfResult(
                name="JitEMD_scaling",
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
# Test 3: JitEMD vs Standard EMD Comparison
# =============================================================================


def test_jit_vs_standard(signal_lengths: List[int] = None, runs: int = 20, warmup: int = 5) -> List[PerfResult]:
    """Compare JitEMD performance against standard EMD."""
    if signal_lengths is None:
        signal_lengths = [500, 1000, 2000, 5000]

    results = []

    # Create instances
    standard_emd = EMD()
    jit_emd = JitEMD()

    # Pre-warm JIT
    warm_signal = generate_test_signal(100, "medium")
    warm_t = np.linspace(0, 1, 100, dtype=np.float64)
    jit_emd.emd(warm_signal, warm_t)

    for n in signal_lengths:
        signal = generate_test_signal(n, "medium")
        t = np.linspace(0, 1, n, dtype=np.float64)

        # Standard EMD
        std_stats = benchmark(standard_emd.emd, signal, runs=runs, warmup=warmup)
        results.append(
            PerfResult(
                name="EMD_comparison",
                params={"signal_length": n, "implementation": "standard"},
                mean=std_stats.mean,
                std=std_stats.std,
                min=std_stats.min,
                max=std_stats.max,
                runs=runs,
                trimmed_mean=std_stats.trimmed_mean,
            )
        )

        # JIT EMD
        jit_stats = benchmark(jit_emd.emd, signal, t, runs=runs, warmup=warmup)
        results.append(
            PerfResult(
                name="EMD_comparison",
                params={"signal_length": n, "implementation": "jit"},
                mean=jit_stats.mean,
                std=jit_stats.std,
                min=jit_stats.min,
                max=jit_stats.max,
                runs=runs,
                trimmed_mean=jit_stats.trimmed_mean,
            )
        )

        # Speedup ratio
        speedup = std_stats.mean / jit_stats.mean if jit_stats.mean > 0 else 0
        results.append(
            PerfResult(
                name="EMD_comparison",
                params={"signal_length": n, "implementation": "speedup"},
                mean=speedup,
                std=0.0,
                min=speedup,
                max=speedup,
                runs=1,
                trimmed_mean=speedup,
                extra={"jit_speedup": f"{speedup:.2f}x"},
            )
        )

    return results


# =============================================================================
# Test 4: JitEMD Spline Methods
# =============================================================================


def test_jit_spline_methods(signal_length: int = 2000, runs: int = 20, warmup: int = 5) -> List[PerfResult]:
    """Compare performance of different spline interpolation methods in JitEMD."""
    # JitEMD supports: cubic, akima (based on code inspection)
    spline_kinds = ["cubic", "akima"]
    signal = generate_test_signal(signal_length, "medium")
    t = np.linspace(0, 1, signal_length, dtype=np.float64)

    results = []

    for spline_kind in spline_kinds:
        jit_emd = JitEMD(spline_kind=spline_kind)

        # Warm up this specific spline
        warm_signal = generate_test_signal(100, "medium")
        warm_t = np.linspace(0, 1, 100, dtype=np.float64)
        try:
            jit_emd.emd(warm_signal, warm_t)
        except Exception as e:
            print(f"  Spline '{spline_kind}' warmup failed: {e}")
            continue

        try:
            stats = benchmark(jit_emd.emd, signal, t, runs=runs, warmup=warmup)
            results.append(
                PerfResult(
                    name="JitEMD_spline",
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
# Test 5: JitEMD Extrema Detection Methods
# =============================================================================


def test_jit_extrema_detection(signal_length: int = 2000, runs: int = 20, warmup: int = 5) -> List[PerfResult]:
    """Compare 'simple' vs 'parabol' extrema detection methods in JitEMD."""
    methods = ["simple", "parabol"]
    signal = generate_test_signal(signal_length, "medium")
    t = np.linspace(0, 1, signal_length, dtype=np.float64)

    results = []

    for method in methods:
        jit_emd = JitEMD(extrema_detection=method)

        # Warm up
        warm_signal = generate_test_signal(100, "medium")
        warm_t = np.linspace(0, 1, 100, dtype=np.float64)
        jit_emd.emd(warm_signal, warm_t)

        stats = benchmark(jit_emd.emd, signal, t, runs=runs, warmup=warmup)
        results.append(
            PerfResult(
                name="JitEMD_extrema",
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
# Test 6: Repeated Execution Benefits
# =============================================================================


def test_repeated_execution(signal_length: int = 1000, iterations: int = 100) -> List[PerfResult]:
    """Test how JIT benefits from repeated execution (amortized compilation cost)."""
    results = []
    signal = generate_test_signal(signal_length, "medium")
    t = np.linspace(0, 1, signal_length, dtype=np.float64)

    # JitEMD - includes compilation in first run
    jit_emd = JitEMD()
    jit_times = []
    for i in range(iterations):
        reset_random_state(DEFAULT_SEED + i)
        start = time.perf_counter()
        jit_emd.emd(signal, t)
        elapsed = time.perf_counter() - start
        jit_times.append(elapsed)

    # Standard EMD
    standard_emd = EMD()
    std_times = []
    for i in range(iterations):
        reset_random_state(DEFAULT_SEED + i)
        start = time.perf_counter()
        standard_emd.emd(signal)
        elapsed = time.perf_counter() - start
        std_times.append(elapsed)

    # Total time comparison
    jit_total = sum(jit_times)
    std_total = sum(std_times)

    # Amortized time (excluding first JIT run)
    jit_amortized = sum(jit_times[1:]) / (iterations - 1) if iterations > 1 else jit_times[0]
    std_amortized = sum(std_times) / iterations

    results.append(
        PerfResult(
            name="repeated_execution",
            params={"signal_length": signal_length, "iterations": iterations, "metric": "total_time"},
            mean=jit_total,
            std=0.0,
            min=jit_total,
            max=jit_total,
            runs=iterations,
            trimmed_mean=jit_total,
            extra={"implementation": "jit", "std_total": std_total},
        )
    )

    results.append(
        PerfResult(
            name="repeated_execution",
            params={"signal_length": signal_length, "iterations": iterations, "metric": "total_time"},
            mean=std_total,
            std=0.0,
            min=std_total,
            max=std_total,
            runs=iterations,
            trimmed_mean=std_total,
            extra={"implementation": "standard"},
        )
    )

    # Speedup for repeated execution
    speedup_total = std_total / jit_total if jit_total > 0 else 0
    speedup_amortized = std_amortized / jit_amortized if jit_amortized > 0 else 0

    results.append(
        PerfResult(
            name="repeated_execution",
            params={"signal_length": signal_length, "iterations": iterations, "metric": "speedup"},
            mean=speedup_total,
            std=0.0,
            min=speedup_total,
            max=speedup_total,
            runs=1,
            trimmed_mean=speedup_amortized,
            extra={
                "total_speedup": f"{speedup_total:.2f}x",
                "amortized_speedup": f"{speedup_amortized:.2f}x",
            },
        )
    )

    return results


# =============================================================================
# Test 7: Signal Complexity Impact on JitEMD
# =============================================================================


def test_jit_signal_complexity(signal_length: int = 2000, runs: int = 20, warmup: int = 5) -> List[PerfResult]:
    """Test how signal complexity affects JitEMD performance."""
    complexities = ["simple", "medium", "complex"]
    results = []

    jit_emd = JitEMD()

    # Pre-warm
    warm_signal = generate_test_signal(100, "medium")
    warm_t = np.linspace(0, 1, 100, dtype=np.float64)
    jit_emd.emd(warm_signal, warm_t)

    for complexity in complexities:
        signal = generate_test_signal(signal_length, complexity)
        t = np.linspace(0, 1, signal_length, dtype=np.float64)

        stats = benchmark(jit_emd.emd, signal, t, runs=runs, warmup=warmup)

        # Get stats from last run
        imfs = jit_emd.emd(signal, t)
        n_imfs = imfs.shape[0]

        results.append(
            PerfResult(
                name="JitEMD_complexity",
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
# Test 8: JIT Function-level Profiling
# =============================================================================


def test_jit_function_breakdown(signal_length: int = 2000, runs: int = 10) -> List[PerfResult]:
    """Profile individual JIT functions to identify bottlenecks."""
    from PyEMD.experimental.jitemd import (
        extract_max_min_extrema,
        find_extrema,
        spline_points,
    )

    results = []
    signal = generate_test_signal(signal_length, "medium")
    t = np.linspace(0, 1, signal_length, dtype=np.float64)

    # Warm up all functions
    jit_emd = JitEMD()
    jit_emd.emd(signal, t)

    config = default_emd_config
    nbsym = int(config["nbsym"])

    # Test find_extrema
    extrema_times = []
    for i in range(runs):
        start = time.perf_counter()
        find_extrema(t, signal, "simple")
        elapsed = time.perf_counter() - start
        extrema_times.append(elapsed)

    stats = BenchmarkStats.from_times(extrema_times)
    results.append(
        PerfResult(
            name="JitEMD_function",
            params={"function": "find_extrema", "signal_length": signal_length},
            mean=stats.mean,
            std=stats.std,
            min=stats.min,
            max=stats.max,
            runs=runs,
            trimmed_mean=stats.trimmed_mean,
        )
    )

    # Test extract_max_min_extrema
    extract_times = []
    for i in range(runs):
        start = time.perf_counter()
        extract_max_min_extrema(t, signal, nbsym, "simple")
        elapsed = time.perf_counter() - start
        extract_times.append(elapsed)

    stats = BenchmarkStats.from_times(extract_times)
    results.append(
        PerfResult(
            name="JitEMD_function",
            params={"function": "extract_max_min_extrema", "signal_length": signal_length},
            mean=stats.mean,
            std=stats.std,
            min=stats.min,
            max=stats.max,
            runs=runs,
            trimmed_mean=stats.trimmed_mean,
        )
    )

    # Test spline_points (need extrema first)
    max_extrema, min_extrema = extract_max_min_extrema(t, signal, nbsym, "simple")

    spline_times = []
    for i in range(runs):
        start = time.perf_counter()
        spline_points(t, max_extrema, "cubic")
        spline_points(t, min_extrema, "cubic")
        elapsed = time.perf_counter() - start
        spline_times.append(elapsed)

    stats = BenchmarkStats.from_times(spline_times)
    results.append(
        PerfResult(
            name="JitEMD_function",
            params={"function": "spline_points", "signal_length": signal_length},
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
    """Run all JitEMD performance tests.

    Args:
        quick: If True, run with smaller parameters for faster feedback
        save: If True, save results to timestamped directory

    Returns:
        List of all performance results
    """
    reset_random_state()

    print("PyEMD JitEMD Performance Test Suite")
    print("=" * 60)

    system_info = get_system_info()
    print(f"Timestamp: {system_info['timestamp']}")
    print(f"Git commit: {system_info['git_commit'][:8]}...")
    print(f"PyEMD version: {system_info['pyemd_version']}")
    print(f"Numba version: {system_info['numba_version']}")

    if quick:
        print("\nRunning in QUICK mode (smaller parameters)")
        signal_lengths = [500, 1000, 2000]
        runs = 10
        warmup = 3
        repeated_iterations = 50
        prefix = "jitemd_quick"
    else:
        print("\nRunning FULL test suite")
        signal_lengths = [500, 1000, 2000, 5000, 10000]
        runs = 20
        warmup = 5
        repeated_iterations = 100
        prefix = "jitemd_full"

    all_results = []

    # Test 1: JIT Warmup
    print("\n[1/8] Testing JIT compilation warmup overhead...")
    results = test_jit_warmup(signal_length=1000, runs=runs)
    print_results(results, "JIT Warmup Overhead")
    all_results.extend(results)

    # Test 2: JitEMD Scaling
    print("[2/8] Testing JitEMD scaling with signal length...")
    results = test_jitemd_scaling(signal_lengths, runs=runs, warmup=warmup)
    print_results(results, "JitEMD Scaling Test")
    all_results.extend(results)

    # Test 3: JitEMD vs Standard EMD
    print("[3/8] Comparing JitEMD vs Standard EMD...")
    results = test_jit_vs_standard(signal_lengths[:4], runs=runs, warmup=warmup)
    print_results(results, "JitEMD vs Standard EMD")
    all_results.extend(results)

    # Test 4: Spline Methods
    print("[4/8] Testing JitEMD spline methods...")
    results = test_jit_spline_methods(signal_length=2000, runs=runs, warmup=warmup)
    print_results(results, "JitEMD Spline Methods")
    all_results.extend(results)

    # Test 5: Extrema Detection
    print("[5/8] Testing JitEMD extrema detection methods...")
    results = test_jit_extrema_detection(signal_length=2000, runs=runs, warmup=warmup)
    print_results(results, "JitEMD Extrema Detection")
    all_results.extend(results)

    # Test 6: Repeated Execution
    print("[6/8] Testing repeated execution benefits...")
    results = test_repeated_execution(signal_length=1000, iterations=repeated_iterations)
    print_results(results, "Repeated Execution Benefits")
    all_results.extend(results)

    # Test 7: Signal Complexity
    print("[7/8] Testing signal complexity impact...")
    results = test_jit_signal_complexity(signal_length=2000, runs=runs, warmup=warmup)
    print_results(results, "Signal Complexity Impact")
    all_results.extend(results)

    # Test 8: Function Breakdown
    print("[8/8] Profiling individual JIT functions...")
    results = test_jit_function_breakdown(signal_length=2000, runs=runs)
    print_results(results, "JIT Function Breakdown")
    all_results.extend(results)

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"Total tests run: {len(all_results)}")

    if save:
        results_dir = create_results_dir(prefix)
        save_results(all_results, results_dir, system_info)

    return all_results


def run_single_test(test_name: str, save: bool = True) -> List[PerfResult]:
    """Run a single test by name.

    Args:
        test_name: One of 'warmup', 'scaling', 'comparison', 'splines',
                   'extrema', 'repeated', 'complexity', 'functions'
        save: If True, save results to timestamped directory

    Returns:
        List of performance results
    """
    reset_random_state()
    system_info = get_system_info()

    test_map = {
        "warmup": (test_jit_warmup, "JIT Warmup Overhead"),
        "scaling": (test_jitemd_scaling, "JitEMD Scaling Test"),
        "comparison": (test_jit_vs_standard, "JitEMD vs Standard EMD"),
        "splines": (test_jit_spline_methods, "JitEMD Spline Methods"),
        "extrema": (test_jit_extrema_detection, "JitEMD Extrema Detection"),
        "repeated": (test_repeated_execution, "Repeated Execution Benefits"),
        "complexity": (test_jit_signal_complexity, "Signal Complexity Impact"),
        "functions": (test_jit_function_breakdown, "JIT Function Breakdown"),
    }

    if test_name not in test_map:
        raise ValueError(f"Unknown test: {test_name}. Choose from: {list(test_map.keys())}")

    func, title = test_map[test_name]
    results = func()
    print_results(results, title)

    if save:
        results_dir = create_results_dir(f"jitemd_{test_name}")
        save_results(results, results_dir, system_info)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PyEMD JitEMD Performance Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python perf_test_jitemd.py                    # Full test suite
  python perf_test_jitemd.py --quick            # Quick test suite
  python perf_test_jitemd.py --test warmup      # Single test
  python perf_test_jitemd.py --test comparison  # Compare JIT vs standard
  python perf_test_jitemd.py --no-save          # Don't save results
  python perf_test_jitemd.py --profile          # Profile the test suite

Results are saved to: perf_test/results/<timestamp>_jitemd_<prefix>/
        """,
    )
    parser.add_argument("--quick", action="store_true", help="Run quick tests with smaller parameters")
    parser.add_argument(
        "--test",
        type=str,
        choices=[
            "warmup",
            "scaling",
            "comparison",
            "splines",
            "extrema",
            "repeated",
            "complexity",
            "functions",
            "all",
        ],
        default="all",
        help="Which test to run (default: all)",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")
    parser.add_argument("--profile", action="store_true", help="Run test suite with cProfile profiling")

    args = parser.parse_args()
    save = not args.no_save

    if args.profile:
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
