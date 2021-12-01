"""Calculate the statistical Significance of IMFs."""
import logging
import math

import numpy as np
from scipy import stats
from scipy.signal import find_peaks


# helper function: Find mean period of an IMF
def mean_period(data):
    """Return mean-period of signal."""
    peaks = len(find_peaks(data, height=0)[0])
    return len(data) / peaks if peaks > 0 else len(data)


# helper function: find energy of signal/IMF
def energy(data):
    """Return energy of signal."""
    return sum(pow(data, 2))


# helper function: find IMF significance in 'a priori' test
def significance_apriori(energy_density, T, N, alpha):
    """Check a priori significance and Return True if significant else False."""
    k = abs(stats.norm.ppf((1 - alpha) / 2))
    upper_limit = -T + (k * (math.sqrt(2 / N) * math.exp(T / 2)))
    lower_limit = -T - (k * (math.sqrt(2 / N) * math.exp(T / 2)))

    return not (lower_limit <= energy_density <= upper_limit)


# helper function: find significance in 'a posteriori' test
def significance_aposteriori(scaled_energy_density, T, N, alpha):
    """Check a posteriori significance and Return True if significant else False."""
    k = abs(stats.norm.ppf((1 - alpha) / 2))
    upper_limit = -T + (k * (math.sqrt(2 / N) * math.exp(T / 2)))
    return not (scaled_energy_density <= upper_limit)


def whitenoise_check(IMFs: np.ndarray, test: str = "aposteriori", rescaling_imf: int = 1, alpha: float = 0.95):
    """Whitenoise statistical significance test.

    References
    ----------
    -- Zhaohua Wu, and Norden E. Huang. “A Study of the Characteristics of White Noise Using the
       Empirical Mode Decomposition Method.” Proceedings: Mathematical, Physical and Engineering
       Sciences, vol. 460, no. 2046, The Royal Society, 2004, pp. 1597–611, http://www.jstor.org/stable/4143111.

    Parameters
    ----------
    IMFs: np.ndarray
        (Required) numpy array containing IMFs computed from a normalized signal
    test: str
        (Optional) It can be used to select other test types like 'apriori'. (default 'aposteriori')
    rescaling_imf: int
        (Optional) ith IMF of the signal used in rescaling for 'a posteriori' test. (default 1)
    alpha:
        (Optional) The percentiles at which the test is to be performed; 0 < alpha < 1; (default 0.95)

    Returns
    -------
    Dictionary with keys as ith IMF,
    Values: 0, indicates IMF is not significant and has noise
            1, indicates IMF is significant and has some information
            None, if input IMFs have NaN, check is skipped

    Examples
    --------
    >>> import numpy as np
    >>> from PyEMD.significancetest import whitenoisecheck
    >>> T = np.linspace(0, 1, 100)
    >>> S = np.sin(2*2*np.pi*T)
    >>> significant_imfs = whitenoisecheck(S, test='apriori')
    >>> significant_imfs
    {1: 0, 2: 1}
    >>> type(significant_imfs)
    <class 'dict'>
    """
    assert 0 < alpha < 1, "alpha value should be in between (0,1)"
    assert test == "apriori" or test == "aposteriori", "Invalid test type"
    assert isinstance(IMFs, np.ndarray), "Invalid Data type, Pass a numpy.ndarray containing IMFs"
    assert rescaling_imf > 0 and rescaling_imf <= len(IMFs), "Invalid rescaling IMF"

    N = len(IMFs[0])
    output = {}
    if N == 0:
        return {}
    if np.isnan(np.sum(IMFs)):
        # Return NaN if input has NaN
        logging.getLogger("PyEMD").warning("Detected NaN values during whitenoise check. Skipping check.")
        return None

    if test == "apriori":
        for idx, imf in enumerate(IMFs):
            log_T = math.log(mean_period(imf))
            energy_density = math.log(energy(imf) / N)
            sig_priori = significance_apriori(energy_density, log_T, N, alpha)

            output[idx + 1] = int(sig_priori)

    elif test == "aposteriori":
        scaling_imf_mean_period = math.log(mean_period(IMFs[rescaling_imf - 1]))
        scaling_imf_energy_density = math.log(energy(IMFs[rescaling_imf - 1]) / N)

        k = abs(stats.norm.ppf((1 - alpha) / 2))
        up_limit = -scaling_imf_mean_period + (k * math.sqrt(2 / N) * math.exp(scaling_imf_mean_period) / 2)

        scaling_factor = up_limit - scaling_imf_energy_density

        for idx, imf in enumerate(IMFs):
            log_T = math.log(mean_period(imf))
            energy_density = math.log(energy(imf) / N)
            scaled_energy_density = energy_density + scaling_factor
            sig_aposteriori = significance_aposteriori(scaled_energy_density, log_T, N, alpha)

            output[idx + 1] = int(sig_aposteriori)

    else:
        raise AssertionError("Only 'apriori' and 'aposteriori' are allowed")

    return output
