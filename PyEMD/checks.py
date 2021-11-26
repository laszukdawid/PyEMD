from PyEMD import EMD, EEMD, CEEMDAN
import numpy as np
from scipy import stats
import math
from scipy.signal import argrelextrema

#helper function: normalize data
def normalize(data):
    return (data - data.mean())/(data.max() - data.min())

#helper function: Find mean period of an IMF
def mean_period(data):
    return len(data)/len(argrelextrema(S, np.greater)[0])

#helper function: find energy of signal/IMF
def energy(data):
    return sum(pow(data, 2))

#helper function: find IMF significance in 'a priori' test
def significance_apriori(energy_density, T, N, alpha):
    k = abs(stats.norm.ppf((1-alpha)/2))
    upper_limit = -math.log(T) + ( k * (math.sqrt(2/N) * math.exp(math.log(T)/2)) )
    lower_limit = -math.log(T) - ( k * (math.sqrt(2/N) * math.exp(math.log(T)/2)) )

    return not (lower_limit <= energy_density <= upper_limit)


#helper function: find significance in 'a posteriori' test
def significance_aposteriori(scaled_energy_density, T, N, alpha):
    k = abs(stats.norm.ppf((1-alpha)/2))
    lower_limit = -math.log(T) + ( k * (math.sqrt(2/N) * math.exp(math.log(T)/2)) )

    return not (scaled_energy_density <= lower_limit)


def whitenoise_check(data: np.ndarray, method: str='EMD',  test: str='aposteriori', rescaling_imf: int=1, alpha: float= 0.95, **kwargs):
    """Returns a dictionary with IMF number(index starts at 1) as key; 0 value means IMF has noise in it and is not statistically significant; 1 value means IMF has some information and is statistically significant

    References
    ----------
    -- Zhaohua Wu, and Norden E. Huang. “A Study of the Characteristics of White Noise Using the Empirical Mode Decomposition Method.” Proceedings: Mathematical, Physical and Engineering Sciences, vol. 460, no. 2046, The Royal Society, 2004, pp. 1597–611, http://www.jstor.org/stable/4143111.
    
    Parameters
    ----------
    data: np.ndarray
        (Required) The original signal/time-series in the form of a numpy.ndarray, Note: Sending decomposed IMFs as an input will not give proper results
    method: str
        (optional) Default value in 'EMD'. it can be used to select other decomposition methods like 'EMD', 'CEEMDAN'
    test: str
        (Optional) Default value is 'aposteriori'. It can be used to select other test types like 'apriori'
    rescaling_imf: int
        (Optional) Default IMF number(index starts at 1) used for scaling in 'a posteriori test' is 1. rescaling_imf = ith IMF of the signal. Basically, if we are aware that a certain IMF is purely composed of noise we can use that rescale all the other IMFs.
    alpha:
        (Optional) Default is 0.95 or 95%. The percentiles at which the test is to be performed; 0 < alpha < 1;  
    **kwargs:
        The keyword arguments for configuring the decomposition method of choice.

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

    assert alpha > 0 and alpha < 1, "alpha value should be in between (0,1)"
    assert method == 'EMD' or method == 'EEMD' or method == 'CEEMDAN', 'Invalid decomposition method'
    assert test == 'apriori' or test == 'aposteriori', 'Invalid test type'
    assert isinstance(data, np.ndarray), 'Invalid Data type, Pass a numpy.ndarray'

    N = len(data)
    if method == 'EMD':
        emd = EMD(**kwargs)
        IMFs = emd.emd(normalize(data))
    elif method == 'EEMD':
        eemd = EEMD(**kwargs)
        IMFs = eemd.eemd(normalize(data))
    elif method == 'CEEMDAN':
        ceemdan = CEEMDAN(**kwargs)
        IMFs = ceemdan.ceemdan(normalize(data))

    assert rescaling_imf > 0 and rescaling_imf <= len(IMFs), 'Invalid rescaling IMF'

    output = {}
    if test == 'apriori':
        for idx,imf in enumerate(IMFs):
            T = mean_period(imf)
            energy_density = energy(imf)/N
            sig_priori = significance_apriori(math.log(energy_density), T, N, alpha)
            if sig_priori:
                output[idx+1] = 1
            else:
                output[idx+1] = 0
    elif test == 'aposteriori':
        scaling_imf_mean_period = mean_period(IMFs[rescaling_imf-1])
        scaling_imf_energy_density = energy(IMFs[rescaling_imf-1])/N

        k = abs(stats.norm.ppf((1-alpha)/2))
        upper_limit = -math.log(scaling_imf_mean_period) + ( k * math.sqrt(2/N)*math.exp(math.log(scaling_imf_mean_period)/2) )

        scaling_factor = upper_limit - math.log(scaling_imf_energy_density)
        for idx,imf in enumerate(IMFs):
            T = mean_period(imf)
            energy_density = energy(imf)/N
            scaled_energy_density = math.log(energy_density) + scaling_factor
            sig_aposteriori = significance_aposteriori(scaled_energy_density, T, N, alpha)
 
            output[idx+1] = int(sig_priori)

    else:
        raise AssertionError("Only 'apriori' and 'aposteriori' are allowed")
    
    return output
