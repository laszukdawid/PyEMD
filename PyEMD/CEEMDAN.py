#!/usr/bin/python
# coding: UTF-8
#
# Author:   Dawid Laszuk
# Contact:  laszukdawid@gmail.com
#
# Feel free to contact for any information.
"""
.. currentmodule:: CEEMDAN
"""

from __future__ import print_function


import logging
import numpy as np

from multiprocessing import Pool

# Python3 handles mutliprocessing much better.
# For Python2 we need to pickle instance differently.
import sys
if sys.version_info[0] < 3:
    import copy_reg as copy_reg
    import types
    def _pickle_method(m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)

    copy_reg.pickle(types.MethodType, _pickle_method)

class CEEMDAN:
    """
    **"Complete Ensemble Empirical Mode Decomposition with Adaptive Noise"**

    "Complete ensemble empirical mode decomposition with adaptive
    noise" (CEEMDAN) [Torres2011]_  is noise-assisted EMD technique.
    Word "complete" presumably refers to decomposing completly
    everything, even added perturbation (noise).

    Provided implementation contains proposed "improvmenets" from
    paper [Colominas2014]_.

    Any parameters can be updated directly on the instance or passed
    through a `configuration` dictionary.

    Goodness of the decomposition can be configured by modifying threshold
    values. Two are `range_thr` and `total_power_thr` which relate to
    the value range (max - min) and check for total power below, respectively.

    Parameters
    ----------

    trials : int (default: 100)
        Number of trials or EMD performance with added noise.

    epsilon : float (default: 0.005)
        Scale for added noise (:math:`\epsilon`) which multiply std :math:`\sigma`:
        :math:`\\beta = \epsilon \cdot \sigma`

    ext_EMD : EMD (default: None)
        One can pass EMD object defined outside, which will be
        used to compute IMF decompositions in each trial. If none
        is passed then EMD with default options is used.

    References
    ----------

    .. [Torres2011] M.E. Torres, M.A. Colominas, G. Schlotthauer, P. Flandrin
        A complete ensemble empirical mode decomposition with adaptive noise.
        Acoustics, Speech and Signal Processing (ICASSP), 2011, pp. 4144--4147

    .. [Colominas2014] M.A. Colominas, G. Schlotthauer, M.E. Torres,
        Improved complete ensemble EMD: A suitable tool for biomedical signal
        processing, In Biomed. Sig. Proc. and Control, V. 14, 2014, pp. 19--29
    """

    logger = logging.getLogger(__name__)

    noise_kinds_all = ["normal", "uniform"]

    def __init__(self, trials=100, epsilon=0.005, ext_EMD=None, parallel=True, **config):
        """
        Configuration can be passed through config dictionary.
        For example, updating threshold would be through:

        >>> config = {"range_thr": 0.001, "total_power_thr": 0.01}
        >>> emd = EMD(**config)
        """

        # Ensemble constants
        self.trials = trials
        self.epsilon = epsilon
        self.all_noise_std = np.zeros(self.trials)

        self.beta_progress = True # Scale noise by std
        self.random = np.random.RandomState()
        self.noise_kind = "normal"
        self.parallel = parallel

        self.all_noise_EMD = []

        if ext_EMD is None:
            from PyEMD import EMD
            self.EMD = EMD()
        else:
            self.EMD = ext_EMD

        self.range_thr = 0.01
        self.total_power_thr = 0.05

        # By default (None) Pool spawns #processes = #CPU
        if parallel:
            processes = None if "processes" not in config else config["processes"]
            self.pool = Pool(processes=processes)

        # Update based on options
        for key in config.keys():
            if key in self.__dict__.keys():
                self.__dict__[key] = config[key]
            elif key in self.EMD.__dict__.keys():
                self.EMD.__dict__[key] = config[key]

    def __call__(self, S, T=None, max_imf=-1):
        return self.ceemdan(S, T=T, max_imf=max_imf)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        if 'pool' in self_dict:
            del self_dict['pool']
        return self_dict

    def generate_noise(self, scale, size):
        """
        Generate noise with specified parameters.
        Currently supported distributions are:

        * *normal* with std equal scale.
        * *uniform* with range [-scale/2, scale/2].

        Parameters
        ----------

        scale : float
            Width for the distribution.
        size : int
            Number of generated samples.

        Returns
        -------

        noise : numpy array
            Noise sampled from selected distribution.
        """

        if self.noise_kind=="normal":
            noise = self.random.normal(loc=0, scale=scale, size=size)
        elif self.noise_kind=="uniform":
            noise = self.random.uniform(low=-scale/2, high=scale/2, size=size)
        else:
            raise ValueError("Unsupported noise kind. Please assigned `noise_kind`"
                + " to be one of these: " + str(self.noise_kinds_all))

        return noise

    def noise_seed(self, seed):
        """Set seed for noise generation."""
        self.random.seed(seed)

    def ceemdan(self, S, T=None, max_imf=-1):

        scale_s = np.std(S)
        S = S/scale_s

        # Define all noise
        self.all_noises = self.generate_noise(1, (self.trials,S.size))

        # Decompose all noise and remember 1st's std
        self.logger.debug("Decomposing all noises")
        for trial, noise in enumerate(self.all_noises):
            _imfs = self.emd(noise, T, max_imf=-1)

            # If beta_progress then scale all IMFs with 1st std
            if self.beta_progress:
                _imfs = _imfs/np.std(_imfs[0])
            self.all_noise_EMD.append(_imfs)

        # Create first IMF
        last_imf = self._eemd(S, T, 1)[0]
        res = np.empty(S.size)

        all_cimfs = last_imf.reshape((-1, last_imf.size))
        prev_res = S - last_imf

        self.logger.debug("Starting CEEMDAN")
        while(True):
            # Check end condition in the beginning because we've already have 1 IMF
            if self.end_condition(S, all_cimfs, max_imf):
                self.logger.debug("End Condition - Pass")
                break

            imfNo = all_cimfs.shape[0]
            beta = self.epsilon*np.std(prev_res)

            local_mean = np.zeros(S.size)
            for trial in range(self.trials):
                # Skip if noise[trial] didn't have k'th mode
                noise_imf = self.all_noise_EMD[trial]
                res = prev_res.copy()
                if len(noise_imf) > imfNo:
                    res += beta*noise_imf[imfNo]

                # Extract local mean, which is at 2nd position
                imfs = self.emd(res, T, 1)
                local_mean += imfs[-1]/self.trials

            last_imf = prev_res - local_mean
            all_cimfs = np.vstack((all_cimfs, last_imf))
            prev_res = local_mean.copy()

        # END of while

        res = S - np.sum(all_cimfs, axis=0)
        all_cimfs = np.vstack((all_cimfs,res))
        all_cimfs = all_cimfs*scale_s

        # Emptyfy all IMFs noise
        del self.all_noise_EMD[:]

        return all_cimfs

    def end_condition(self, S, cIMFs, max_imf):
        """Test for end condition of CEEMDAN.

        Procedure stops if:

        * number of components reach provided `max_imf`, or
        * last component is close to being pure noise (range or power), or
        * set of provided components reconstructs sufficiently input.

        Parameters
        ----------
        S : numpy array
            Original signal on which CEEMDAN was performed.
        cIMFs : numpy 2D array
            Set of cIMFs where each row is cIMF.

        Returns
        -------
        end : bool
            Whether to stop CEEMDAN.
        """
        imfNo = cIMFs.shape[0]

        # Check if hit maximum number of cIMFs
        if max_imf > 0 and imfNo >= max_imf:
            return True

        # Compute EMD on residue
        R = S - np.sum(cIMFs, axis=0)
        _test_imf = self.emd(R, None, max_imf=1)

        # Check if residue is IMF or no extrema
        if _test_imf.shape[0] == 1:
            self.logger.debug("Not enough extrema")
            return True

        # Check for range threshold
        if np.max(R) - np.min(R) < self.range_thr:
            self.logger.debug("FINISHED -- RANGE")
            return True

        # Check for power threshold
        if np.sum(np.abs(R)) < self.total_power_thr:
            self.logger.debug("FINISHED -- SUM POWER")
            return True

        return False

    def _eemd(self, S, T=None, max_imf=-1):
        if T is None: T = np.arange(len(S), dtype=S.dtype)

        self._S = S
        self._T = T
        self._N = N = len(S)
        self.max_imf = max_imf

        # For trial number of iterations perform EMD on a signal
        # with added white noise
        _map = self.pool.map if self.parallel else map
        all_IMFs = _map(self._trial_update, range(self.trials))

        max_imfNo = max([IMFs.shape[0] for IMFs in all_IMFs])

        self.E_IMF = np.zeros((max_imfNo, N))
        for IMFs in all_IMFs:
            self.E_IMF[:IMFs.shape[0]] += IMFs

        return self.E_IMF/self.trials

    def _trial_update(self, trial):
        # Generate noise
        noise = self.epsilon*self.all_noise_EMD[trial][0]
        return self.emd(self._S+noise, self._T, self.max_imf)

    def emd(self, S, T, max_imf=-1):
        """Vanilla EMD method.

        Provides emd evaluation from provided EMD class.
        For reference please see :class:`PyEMD.EMD`.
        """
        return self.EMD.emd(S, T, max_imf)

###################################################
## Beginning of program

if __name__ == "__main__":

    import pylab as plt

    # Logging options
    logging.basicConfig(level=logging.INFO)

    max_imf = -1

    # Signal options
    N = 500
    tMin, tMax = 0, 2*np.pi
    T = np.linspace(tMin, tMax, N)

    S = 3*np.sin(4*T) + 4*np.cos(9*T) + np.sin(8.11*T+1.2)

    # Prepare and run EEMD
    trials = 20
    ceemdan = CEEMDAN(trials=trials)

    C_IMFs = ceemdan(S, T, max_imf)
    imfNo  = C_IMFs.shape[0]

    # Plot results in a grid
    c = np.floor(np.sqrt(imfNo+2))
    r = np.ceil((imfNo+2)/c)

    plt.ioff()
    plt.subplot(r,c,1)
    plt.plot(T, S, 'r')
    plt.xlim((tMin, tMax))
    plt.title("Original signal")

    plt.subplot(r,c,2)
    plt.plot(T, S-np.sum(C_IMFs, axis=0), 'r')
    plt.xlim((tMin, tMax))
    plt.title("Residuum")

    for num in range(imfNo):
        plt.subplot(r,c,num+3)
        plt.plot(T, C_IMFs[num],'g')
        plt.xlim((tMin, tMax))
        plt.title("Imf "+str(num+1))

    plt.show()
