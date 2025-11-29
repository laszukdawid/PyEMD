# PyEMD Repository Investigation Summary

## Date: November 29, 2024

## What This Repository Does

**PyEMD** is a Python implementation of **Empirical Mode Decomposition (EMD)** and its variations. EMD is a signal processing technique for decomposing non-linear and non-stationary time series data into simpler components called Intrinsic Mode Functions (IMFs).

### Purpose and Applications

EMD is used for:
- **Signal Analysis**: Decomposing complex signals into simpler oscillatory components
- **Time-Frequency Analysis**: Understanding how signal frequencies change over time
- **Non-stationary Signal Processing**: Analyzing signals where statistical properties change over time
- **Scientific Research**: Applications in climate science, biomedical engineering, finance, and more

### Key Features

The repository provides several EMD variations:

1. **EMD (Empirical Mode Decomposition)**: The core algorithm that decomposes signals into IMFs
2. **EEMD (Ensemble EMD)**: A noise-assisted technique that's more robust than vanilla EMD
3. **CEEMDAN (Complete Ensemble EMD with Adaptive Noise)**: An improved ensemble method
4. **EMD2D/BEMD**: Experimental 2D image decomposition (not fully supported)
5. **JitEMD**: Just-in-time compiled version for performance on large signals

### Customization Options

- **Multiple spline types**: Natural cubic (default), pointwise cubic, Hermite, Akima, PChip, linear
- **Stopping criteria**: Cauchy convergence (default), fixed iterations, consecutive proto-IMFs
- **Extrema detection**: Discrete extrema (default), parabolic interpolation

### Package Information

- **Package Name**: EMD-signal (on PyPI and Conda)
- **Import Name**: PyEMD
- **Version**: 1.6.4
- **License**: Apache-2.0
- **Dependencies**: numpy, scipy, pathos, tqdm

## Bugs Found and Fixed

### 1. **CRITICAL BUG: Root-level `__init__.py` causing test import failures**

**Location**: `/home/runner/work/PyEMD/PyEMD/__init__.py` (repository root)

**Impact**: HIGH - 59 out of 95 tests were failing

**Description**: 
An empty `__init__.py` file existed at the repository root (same level as the PyEMD package directory). When pytest ran from the repository root, it added the parent directory to `sys.path`, which caused Python to treat the repository root as a namespace package named "PyEMD". This created a naming conflict with the actual PyEMD package located in the `PyEMD/` subdirectory.

**Symptoms**:
```python
from PyEMD import EMD
emd = EMD()  # TypeError: 'module' object is not callable
```

The import statement was importing the PyEMD.EMD module instead of the PyEMD.EMD.EMD class.

**Root Cause**:
```
Repository structure:
/home/runner/work/PyEMD/PyEMD/
├── __init__.py              # <- This empty file caused the issue
├── PyEMD/                   # <- Actual package
│   ├── __init__.py          # <- Package's real __init__.py
│   ├── EMD.py               # <- Module containing EMD class
│   └── ...
```

When pytest added `/home/runner/work/PyEMD` to sys.path, importing `PyEMD` would resolve to the root directory, making `PyEMD.EMD` import the module instead of the class.

**Fix**: Removed the root-level `__init__.py` file

**Result**: All 96 tests now pass (21 skipped due to missing optional dependencies)

---

### 2. **Deprecated scipy imports**

**Location**: `PyEMD/EMD2d.py` lines 17-18

**Impact**: MEDIUM - Causes DeprecationWarnings, will break in scipy 2.0

**Description**:
The code used deprecated scipy import paths:
```python
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
```

These submodules are deprecated and scheduled for removal in scipy 2.0.

**Fix**: Updated to use the new import paths:
```python
from scipy.ndimage import maximum_filter
from scipy.ndimage import binary_erosion, generate_binary_structure
```

---

### 3. **Invalid escape sequences in docstrings**

**Location**: 
- `PyEMD/CEEMDAN.py` line 63
- `PyEMD/EEMD.py` lines 40, 42

**Impact**: LOW - Causes SyntaxWarnings but doesn't affect functionality

**Description**:
LaTeX math expressions in docstrings used single backslashes for special characters:
```python
# CEEMDAN.py line 63
:math:`\\beta = \epsilon \cdot \sigma`  # \e is invalid escape sequence

# EEMD.py line 40
:math:`\hat\sigma`  # \h is invalid escape sequence
```

**Fix**: Properly escaped backslashes in raw strings:
```python
# CEEMDAN.py
:math:`\\\\beta = \\epsilon \\cdot \\sigma`

# EEMD.py  
:math:`\\hat\\sigma = \\sigma\\cdot|\\max(S)-\\min(S)|`
```

---

### 4. **Typos in error message**

**Location**: `PyEMD/EMD2d.py` line 22

**Impact**: LOW - Cosmetic issue in error message

**Description**:
Error message contained two typos:
```python
"Required depdenecies are in `requriements-extra`."
```

**Fix**: Corrected spelling:
```python
"Required dependencies are in `requirements-extra`."
```

---

## Potential Issues (Not Fixed - Require Further Investigation)

### 1. **Possible formula error in whitenoise check (checks.py)**

**Location**: `PyEMD/checks.py` lines 119 and 128

**Description**:
Two potentially questionable formulas in the a posteriori whitenoise significance test:

**Line 119**:
```python
up_limit = -scaling_imf_mean_period + (k * math.sqrt(2 / N) * math.exp(scaling_imf_mean_period) / 2)
```
The division by 2 at the end is ambiguous. Based on typical statistical formulas, it might should be:
```python
up_limit = -scaling_imf_mean_period + (k * math.sqrt(2 / N) * math.exp(scaling_imf_mean_period / 2))
```

**Line 128**:
```python
if idx != rescaling_imf - 1:
    scaled_energy_density = math.log((energy(imf) / N) / scaling_factor)
else:
    scaled_energy_density = math.log(scaling_factor)  # <- Inconsistent
```

When processing the rescaling IMF itself, the formula is completely different from other IMFs. This might be intentional for the statistical test, but seems inconsistent.

**Status**: NOT FIXED - All tests pass, and without access to the Wu & Huang 2004 paper referenced in the code, I cannot verify if these are bugs or intentional. The implementation needs verification against the original paper.

---

## Testing Results

**Before fixes**:
- 36 tests passed
- 59 tests failed (due to import issue)
- 21 tests skipped
- Multiple deprecation warnings

**After fixes**:
- 96 tests passed ✓
- 0 tests failed ✓
- 21 tests skipped (due to missing optional dependencies like scikit-image)
- No warnings ✓

## Code Quality Observations

### Strengths:
1. **Well-documented**: Comprehensive docstrings with references to academic papers
2. **Good test coverage**: 117 test cases covering various scenarios
3. **Modular design**: Clean separation between EMD variants
4. **Flexible**: Multiple configuration options for different use cases
5. **Active maintenance**: Recent updates and good issue tracking

### Areas for Improvement:
1. **Root directory cleanup**: The presence of the problematic `__init__.py` suggests inconsistent packaging practices
2. **Dependency management**: Some features require optional dependencies that aren't well-isolated
3. **Formula verification**: Mathematical implementations should include test cases that verify correctness against known results, not just type checking

## Recommendations

1. **Keep the root directory clean**: Ensure no Python files at the repository root level that could conflict with the package
2. **Update scipy imports**: The deprecated imports have been fixed but should be monitored for other deprecated API usage
3. **Add integration tests**: Include tests that verify mathematical correctness against published results
4. **Document optional features**: Better documentation about which features require which optional dependencies
5. **Formula verification**: Have someone familiar with the Wu & Huang 2004 paper verify the whitenoise check implementation

## Summary

PyEMD is a well-maintained, scientifically-grounded signal processing library for Empirical Mode Decomposition. The critical bug (root `__init__.py`) has been fixed, along with several minor issues. The library is now in good working order with all tests passing. One potential mathematical issue in the whitenoise check requires expert review but doesn't prevent the library from functioning correctly.
