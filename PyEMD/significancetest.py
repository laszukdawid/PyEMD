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
    k = abs(stats.norm.ppf((1-alpha)/2))
    if (energy_density <= (-math.log(T) + k*(math.sqrt(2/N)*math.exp(math.log(T)/2)))) and (energy_density >= (-math.log(T) - k*(math.sqrt(2/N)*math.exp(math.log(T)/2)))):
        return False
    else:
        return True

#helper function: find significance in 'a posteriori' test
def significance_aposteriori(scaled_energy_density, T, N, alpha):
    k = abs(stats.norm.ppf((1-alpha)/2))
    if (scaled_energy_density <= (-math.log(T) + k*(math.sqrt(2/N)*math.exp(math.log(T)/2)))):
        return False
    else:
        return True

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

test_data = np.array([1612.099955, 1872.499974, 1349.199993, 1849.799958, 2187.79993, 2747.900004, 2028.199993, 1883.499968, 2353.399962, 2346.49999, 2385.499986, 1735.799965, 2262.299981, 2475.69998, 2832.399969, 1627.79998, 1832.999984, 2223.8, 1852.299986, 2398.800005, 2460.600006, 1721.599968, 2164.699953, 2297.599986, 2116.199967, 1810.199969, 2015.599941, 2246.099977, 2359.599986, 2476.099909, 1854.999945, 1848.80001, 1738.799932, 2429.999947, 1352.399964, 2827.39998, 1691.899996, 2187.247636, 2055.670849, 2476.227106, 2894.536279, 2223.016297, 1667.800095, 1593.206414, 2080.420102, 2392.493201, 2525.678054, 2359.714593, 2764.414422, 2054.748585, 2370.955795, 2574.509022, 3123.112617, 2208.499965, 2427.437107, 2888.377844, 1839.610973, 1644.790258, 2583.974357, 2554.012727, 2688.180162, 937.7219929, 2494.624529, 2889.377336, 2380.749196, 1881.62625, 1757.290773, 1964.164185, 2676.664096, 2646.20634, 2466.383096, 1524.450419, 2655.081313, 2765.296026, 2693.869523, 2999.969905, 2827.312295, 2399.215457, 2220.22048, 1998.112353, 1898.426222, 1446.748416, 3136.590134, 2968.556433, 1983.498138, 2575.42455, 3125.803186, 4573.776578, 3234.599062, 4797.722408, 5072.706734, 2407.703483, 5500.870556, 2581.111854, 3414.67519, 2368.143402, 3202.104436, 3353.771535, 2620.371734, 2795.362439, 2472.294985, 2663.27541, 2853.170998, 2471.682001, 1760.751243, 1866.067174, 3323.570975, 2396.070515, 2223.334293, 2062.451717, 1742.005987, 2125.159392, 1905.346065, 2117.524309, 2966.14579, 2237.294325, 3393.220207, 2909.662327, 2587.688155, 2433.188114])

whitenoisecheck(test_data, method='EEMD')
whitenoisecheck(test_data, test='apriori')

whitenoisecheck(test_data)
whitenoisecheck(test_data, test='apriori')

whitenoisecheck(test_data, method="CEEMDAN")
whitenoisecheck(test_data, test='apriori')