from PyEMD import EMD, EEMD, CEEMDAN
import numpy as np
from scipy import stats
import math


#helper function: normalize data
def normalize(data):
    mean = sum(data)/len(data)
    dmin = min(data)
    dmax = max(data)
    ret = np.array([])
    for val in data:
        ret = np.append(ret, (val-mean)/(dmax-dmin))
    return ret

#helper function: find zero-crossings in IMF, used for finding mean period of IMF
def sign_change(data):
    l = len(data)
    counter=0
    for x in range(l-1):
        if (data[x]>0 or data[x] == 0) and data[x+1] < 0:
            counter+=1
    return counter

#helper function: Find mean period of an IMF
def mean_period(data):
    s = sign_change(data)
    if s != 0:
        return len(data)/s
    else:
        return len(data)


#helper function: find energy of signal/IMF
def energy(data):
    return sum(pow(data, 2))

#helper function: find IMF significance in 'a priori' test
def significance_apriori(energy_density, T, N, alpha):
    try:
        k = abs(stats.norm.ppf((1-alpha)/2))
        if (energy_density <= (-math.log(T) + k*(math.sqrt(2/N)*math.exp(math.log(T)/2)))) and (energy_density >= (-math.log(T) - k*(math.sqrt(2/N)*math.exp(math.log(T)/2)))):
            return False
        else:
            return True
    except Exception as e:
        return False

#helper function: find significance in 'a posteriori' test
def significance_aposteriori(scaled_energy_density, T, N, alpha):
    try:
        k = abs(stats.norm.ppf((1-alpha)/2))
        if (scaled_energy_density <= (-math.log(T) + k*(math.sqrt(2/N)*math.exp(math.log(T)/2)))):
            return False
        else:
            return True
    except Exception as e:
        print(e)
        return False

def whitenoisecheck(data: np.ndarray, method: str='EMD',  test: str='aposteriori', rescaling_imf: int=1, alpha: float= 0.95):
    '''
    References
    --------------
    -- Zhaohua Wu, and Norden E. Huang. “A Study of the Characteristics of White Noise Using the Empirical Mode Decomposition Method.” Proceedings: Mathematical, Physical and Engineering Sciences, vol. 460, no. 2046, The Royal Society, 2004, pp. 1597–611, http://www.jstor.org/stable/4143111.

    Examples
    --------------
    >>> import numpy as np
    >>> from PyEMD.significancetest import whitenoisecheck
    >>> T = np.linspace(0, 1, 100)
    >>> S = np.sin(2*2*np.pi*T)
    >>> significant_imfs = whitenoisecheck(S, test='apriori')
    >>> significant_imfs
    {1: 0, 2: 1}
    >>> type(significant_imfs)
    <class 'dict'>
    '''
    assert alpha > 0 and alpha < 1, "alpha value should be in between (0,1)"
    assert method == 'EMD' or method == 'EEMD' or method == 'CEEMDAN', 'Invalid decomposition method'
    assert test == 'apriori' or test == 'aposteriori', 'Invalid test type'
    assert isinstance(data, np.ndarray), 'Invalid Data type, Pass a numpy.ndarray'

    N = len(data)
    if method == 'EMD':
        emd = EMD()
        IMFs = emd.emd(normalize(data))
    elif method == 'EEMD':
        eemd = EEMD()
        IMFs = eemd.eemd(normalize(data))
    elif method == 'CEEMDAN':
        ceemdan = CEEMDAN()
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
    else:
        scaling_imf_mean_period = mean_period(IMFs[rescaling_imf-1])
        scaling_imf_energy_density = energy(IMFs[rescaling_imf-1])/N
        
        k = abs(stats.norm.ppf((1-alpha)/2))
        upper_limit = -math.log(scaling_imf_mean_period)+k*(math.sqrt(2/N))*math.exp(math.log(scaling_imf_mean_period)/2)
        scaling_factor = upper_limit - math.log(scaling_imf_energy_density)
        
        for idx,imf in enumerate(IMFs):
            T = mean_period(imf)
            energy_density = energy(imf)/N
            scaled_energy_density = math.log(energy_density) + scaling_factor
            sig_aposteriori = significance_aposteriori(scaled_energy_density, T, N, alpha)
            if sig_aposteriori:
                output[idx+1] = 1
            else:
                output[idx+1] = 0
    return output
