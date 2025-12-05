#!/usr/bin/env python
"""
Compare two PyEMD performance test results.

Usage:
    python compare_results.py <baseline> <comparison>
    python compare_results.py results/20241201_120000_full results/20241202_120000_full

Options:
    --threshold PERCENT  Highlight changes greater than this percentage (default: 5)
    --format FORMAT      Output format: text, json, markdown (default: text)
    --alpha FLOAT        Significance level for t-test (default: 0.05)
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def welch_ttest(mean1: float, std1: float, n1: int, mean2: float, std2: float, n2: int) -> Tuple[float, float]:
    """
    Perform Welch's t-test for two samples with different variances.

    Returns (t_statistic, p_value).
    Uses approximation for p-value calculation without scipy.
    """
    if std1 == 0 and std2 == 0:
        # No variance - can't compute t-test
        return 0.0, 1.0

    # Standard error of difference
    se1 = (std1 ** 2) / n1 if n1 > 0 else 0
    se2 = (std2 ** 2) / n2 if n2 > 0 else 0
    se_diff = math.sqrt(se1 + se2)

    if se_diff == 0:
        return 0.0, 1.0

    # t-statistic
    t_stat = (mean1 - mean2) / se_diff

    # Welch-Satterthwaite degrees of freedom
    if se1 + se2 == 0:
        df = 1
    else:
        num = (se1 + se2) ** 2
        denom = (se1 ** 2) / (n1 - 1) if n1 > 1 else 0
        denom += (se2 ** 2) / (n2 - 1) if n2 > 1 else 0
        df = num / denom if denom > 0 else 1

    # Approximate p-value using normal distribution for large df
    # For more accurate results with small df, would need scipy
    # This is a reasonable approximation for df > 30
    if df > 30:
        # Use normal approximation
        z = abs(t_stat)
        # Approximation of 2-tailed p-value from z-score
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    else:
        # For smaller df, use a conservative estimate
        # This is less accurate but avoids scipy dependency
        z = abs(t_stat) * math.sqrt(df / (df + t_stat ** 2))
        p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))

    return t_stat, p_value


@dataclass
class ComparisonResult:
    """Result of comparing two benchmark results."""
    test_name: str
    params: Dict
    baseline_mean: float
    comparison_mean: float
    baseline_trimmed_mean: float  # 10% trimmed mean (used for comparison)
    comparison_trimmed_mean: float
    baseline_std: float
    comparison_std: float
    baseline_runs: int
    comparison_runs: int
    diff_seconds: float  # Based on trimmed mean
    diff_percent: float  # Based on trimmed mean
    is_faster: bool
    is_significant: bool  # Based on statistical test
    p_value: float  # From t-test (uses regular mean/std)
    baseline_cv: float  # Coefficient of variation
    comparison_cv: float

    def __str__(self) -> str:
        direction = "faster" if self.is_faster else "slower"
        sign = "-" if self.is_faster else "+"
        sig_marker = "*" if self.is_significant else ""
        return (
            f"{self.test_name} ({self.params}): "
            f"{self.baseline_trimmed_mean:.4f}s → {self.comparison_trimmed_mean:.4f}s "
            f"({sign}{abs(self.diff_percent):.1f}% {direction}{sig_marker}, p={self.p_value:.3f})"
        )


def load_results(path: Path) -> Tuple[Dict, List[Dict]]:
    """Load results from a results directory or JSON file."""
    if path.is_dir():
        json_path = path / "results.json"
    else:
        json_path = path

    if not json_path.exists():
        raise FileNotFoundError(f"Results file not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    return data.get("system_info", {}), data.get("results", [])


def make_key(result: Dict) -> str:
    """Create a unique key for a result based on name and params."""
    params_str = json.dumps(result["params"], sort_keys=True)
    return f"{result['name']}:{params_str}"


def compare_results(
    baseline: List[Dict],
    comparison: List[Dict],
    threshold_percent: float = 5.0,
    alpha: float = 0.05,
) -> List[ComparisonResult]:
    """Compare two sets of results using statistical significance testing.

    Args:
        baseline: List of baseline benchmark results
        comparison: List of comparison benchmark results
        threshold_percent: Minimum percentage change to consider (in addition to p-value)
        alpha: Significance level for t-test (default 0.05 = 95% confidence)
    """
    # Index baseline by key
    baseline_map = {make_key(r): r for r in baseline}
    comparison_map = {make_key(r): r for r in comparison}

    results = []

    # Find matching tests
    all_keys = set(baseline_map.keys()) | set(comparison_map.keys())

    for key in sorted(all_keys):
        base = baseline_map.get(key)
        comp = comparison_map.get(key)

        if base is None or comp is None:
            continue  # Skip tests that don't exist in both

        base_mean = base["mean"]
        comp_mean = comp["mean"]
        # Use trimmed mean for comparison (falls back to mean for older results)
        base_trimmed = base.get("trimmed_mean", base_mean)
        comp_trimmed = comp.get("trimmed_mean", comp_mean)
        base_std = base.get("std", 0)
        comp_std = comp.get("std", 0)
        base_runs = base.get("runs", 1)
        comp_runs = comp.get("runs", 1)

        # Use trimmed mean for diff calculation (more robust to outliers)
        diff_seconds = comp_trimmed - base_trimmed
        diff_percent = (diff_seconds / base_trimmed) * 100 if base_trimmed > 0 else 0

        # Coefficient of variation (std/mean as percentage) - still uses regular mean
        base_cv = (base_std / base_mean * 100) if base_mean > 0 else 0
        comp_cv = (comp_std / comp_mean * 100) if comp_mean > 0 else 0

        # Perform Welch's t-test (uses regular mean/std for statistical test)
        _, p_value = welch_ttest(base_mean, base_std, base_runs, comp_mean, comp_std, comp_runs)

        # Significant if:
        # 1. p-value is below alpha (statistically significant)
        # 2. AND the percentage change exceeds threshold (practically significant)
        is_significant = p_value < alpha and abs(diff_percent) >= threshold_percent

        results.append(ComparisonResult(
            test_name=base["name"],
            params=base["params"],
            baseline_mean=base_mean,
            comparison_mean=comp_mean,
            baseline_trimmed_mean=base_trimmed,
            comparison_trimmed_mean=comp_trimmed,
            baseline_std=base_std,
            comparison_std=comp_std,
            baseline_runs=base_runs,
            comparison_runs=comp_runs,
            diff_seconds=diff_seconds,
            diff_percent=diff_percent,
            is_faster=diff_seconds < 0,
            is_significant=is_significant,
            p_value=p_value,
            baseline_cv=base_cv,
            comparison_cv=comp_cv,
        ))

    return results


def format_text(
    results: List[ComparisonResult],
    baseline_info: Dict,
    comparison_info: Dict,
    threshold: float,
    alpha: float = 0.05,
) -> str:
    """Format comparison results as plain text."""
    lines = []
    lines.append("=" * 70)
    lines.append(" PyEMD Performance Comparison")
    lines.append("=" * 70)
    lines.append("")

    # System info comparison
    lines.append("Baseline:")
    lines.append(f"  Timestamp: {baseline_info.get('timestamp', 'unknown')}")
    lines.append(f"  Git commit: {baseline_info.get('git_commit', 'unknown')[:8]}...")
    lines.append(f"  PyEMD version: {baseline_info.get('pyemd_version', 'unknown')}")
    lines.append("")

    lines.append("Comparison:")
    lines.append(f"  Timestamp: {comparison_info.get('timestamp', 'unknown')}")
    lines.append(f"  Git commit: {comparison_info.get('git_commit', 'unknown')[:8]}...")
    lines.append(f"  PyEMD version: {comparison_info.get('pyemd_version', 'unknown')}")
    lines.append("")

    lines.append("Statistical parameters:")
    lines.append(f"  Minimum % change threshold: {threshold}%")
    lines.append(f"  Significance level (alpha): {alpha}")
    lines.append(f"  A result is significant if: p < {alpha} AND |change| >= {threshold}%")
    lines.append("  Comparison uses 10% trimmed mean (outlier-robust)")
    lines.append("")

    # Summary
    faster = [r for r in results if r.is_faster and r.is_significant]
    slower = [r for r in results if not r.is_faster and r.is_significant]
    unchanged = [r for r in results if not r.is_significant]

    # Calculate average CV to show measurement quality
    avg_base_cv = sum(r.baseline_cv for r in results) / len(results) if results else 0
    avg_comp_cv = sum(r.comparison_cv for r in results) / len(results) if results else 0
    max_cv = max(max(r.baseline_cv, r.comparison_cv) for r in results) if results else 0

    lines.append("-" * 70)
    lines.append(f" Summary: {len(faster)} faster, {len(slower)} slower, {len(unchanged)} unchanged")
    lines.append(f" Average coefficient of variation: baseline={avg_base_cv:.1f}%, comparison={avg_comp_cv:.1f}%")
    lines.append("-" * 70)

    # Warning about high variance
    if max_cv > 15:
        lines.append("")
        lines.append("WARNING: High variance detected (CV > 15%)!")
        lines.append("Results may be unreliable. Consider:")
        lines.append("  - Closing other applications")
        lines.append("  - Disabling CPU frequency scaling (performance governor)")
        lines.append("  - Running more iterations")
        lines.append("  - Using a dedicated benchmark machine")

    lines.append("")

    # Significant improvements (faster)
    if faster:
        lines.append("FASTER (statistically significant improvements):")
        lines.append("-" * 40)
        for r in sorted(faster, key=lambda x: x.diff_percent):
            lines.append(f"  ✓ {r}")
        lines.append("")

    # Significant regressions (slower)
    if slower:
        lines.append("SLOWER (statistically significant regressions):")
        lines.append("-" * 40)
        for r in sorted(slower, key=lambda x: -x.diff_percent):
            lines.append(f"  ✗ {r}")
        lines.append("")

    # Unchanged
    if unchanged:
        lines.append("NOT SIGNIFICANT (p >= alpha or change < threshold):")
        lines.append("-" * 40)
        for r in unchanged:
            sign = "-" if r.is_faster else "+"
            lines.append(
                f"  = {r.test_name} ({r.params}): "
                f"{r.baseline_trimmed_mean:.4f}s → {r.comparison_trimmed_mean:.4f}s "
                f"({sign}{abs(r.diff_percent):.1f}%, p={r.p_value:.3f}, CV={r.baseline_cv:.1f}%/{r.comparison_cv:.1f}%)"
            )
        lines.append("")

    return "\n".join(lines)


def format_markdown(
    results: List[ComparisonResult],
    baseline_info: Dict,
    comparison_info: Dict,
    threshold: float
) -> str:
    """Format comparison results as markdown."""
    lines = []
    lines.append("# PyEMD Performance Comparison")
    lines.append("")

    # System info
    lines.append("## Environment")
    lines.append("")
    lines.append("| | Baseline | Comparison |")
    lines.append("|---|---|---|")
    lines.append(f"| Timestamp | {baseline_info.get('timestamp', 'unknown')} | {comparison_info.get('timestamp', 'unknown')} |")
    lines.append(f"| Git commit | `{baseline_info.get('git_commit', 'unknown')[:8]}` | `{comparison_info.get('git_commit', 'unknown')[:8]}` |")
    lines.append(f"| PyEMD version | {baseline_info.get('pyemd_version', 'unknown')} | {comparison_info.get('pyemd_version', 'unknown')} |")
    lines.append("")

    # Summary
    faster = [r for r in results if r.is_faster and r.is_significant]
    slower = [r for r in results if not r.is_faster and r.is_significant]
    unchanged = [r for r in results if not r.is_significant]

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **{len(faster)}** tests faster (improvements)")
    lines.append(f"- **{len(slower)}** tests slower (regressions)")
    lines.append(f"- **{len(unchanged)}** tests unchanged (within {threshold}% threshold)")
    lines.append("")
    lines.append("*Note: Comparison uses 10% trimmed mean for outlier robustness*")
    lines.append("")

    # Results table
    lines.append("## Detailed Results")
    lines.append("")
    lines.append("| Test | Params | Baseline | Comparison | Change |")
    lines.append("|------|--------|----------|------------|--------|")

    for r in sorted(results, key=lambda x: x.diff_percent):
        params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
        sign = "" if r.is_faster else "+"
        status = "🟢" if r.is_faster and r.is_significant else ("🔴" if not r.is_faster and r.is_significant else "⚪")
        lines.append(
            f"| {r.test_name} | {params_str} | "
            f"{r.baseline_trimmed_mean:.4f}s | "
            f"{r.comparison_trimmed_mean:.4f}s | "
            f"{status} {sign}{r.diff_percent:.1f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def format_json(
    results: List[ComparisonResult],
    baseline_info: Dict,
    comparison_info: Dict,
    threshold: float
) -> str:
    """Format comparison results as JSON."""
    data = {
        "baseline_info": baseline_info,
        "comparison_info": comparison_info,
        "threshold_percent": threshold,
        "summary": {
            "total": len(results),
            "faster": len([r for r in results if r.is_faster and r.is_significant]),
            "slower": len([r for r in results if not r.is_faster and r.is_significant]),
            "unchanged": len([r for r in results if not r.is_significant]),
        },
        "results": [
            {
                "test_name": r.test_name,
                "params": r.params,
                "baseline_mean": r.baseline_mean,
                "baseline_std": r.baseline_std,
                "baseline_trimmed_mean": r.baseline_trimmed_mean,
                "comparison_mean": r.comparison_mean,
                "comparison_std": r.comparison_std,
                "comparison_trimmed_mean": r.comparison_trimmed_mean,
                "diff_seconds": r.diff_seconds,
                "diff_percent": r.diff_percent,
                "is_faster": r.is_faster,
                "is_significant": r.is_significant,
                "p_value": r.p_value,
            }
            for r in results
        ],
    }
    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two PyEMD performance test results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_results.py results/20241201_full results/20241202_full
  python compare_results.py baseline.json comparison.json --threshold 10
  python compare_results.py old/ new/ --format markdown > comparison.md
  python compare_results.py old/ new/ --alpha 0.01  # stricter significance
        """
    )
    parser.add_argument(
        "baseline",
        type=Path,
        help="Path to baseline results (directory or JSON file)"
    )
    parser.add_argument(
        "comparison",
        type=Path,
        help="Path to comparison results (directory or JSON file)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Minimum percentage change to consider significant (default: 5)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for t-test (default: 0.05 = 95%% confidence)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    try:
        baseline_info, baseline_results = load_results(args.baseline)
        comparison_info, comparison_results = load_results(args.comparison)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not baseline_results:
        print(f"Error: No results found in baseline: {args.baseline}", file=sys.stderr)
        sys.exit(1)

    if not comparison_results:
        print(f"Error: No results found in comparison: {args.comparison}", file=sys.stderr)
        sys.exit(1)

    results = compare_results(baseline_results, comparison_results, args.threshold, args.alpha)

    if not results:
        print("Error: No matching tests found between baseline and comparison", file=sys.stderr)
        sys.exit(1)

    if args.format == "text":
        output = format_text(results, baseline_info, comparison_info, args.threshold, args.alpha)
    elif args.format == "markdown":
        output = format_markdown(results, baseline_info, comparison_info, args.threshold)
    elif args.format == "json":
        output = format_json(results, baseline_info, comparison_info, args.threshold)

    print(output)

    # Exit with non-zero status if there are significant regressions
    regressions = [r for r in results if not r.is_faster and r.is_significant]
    if regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
