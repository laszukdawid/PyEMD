#!/usr/bin/python
# coding: UTF-8
#
# Author:   Dawid Laszuk
# Contact:  laszukdawid@gmail.com
#
# Feel free to contact for any information.
"""
.. currentmodule:: EEMD
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

class EEMD:
    """
    **Ensemble Empirical Mode Decomposition**

    Ensemble empirical mode decomposition (EEMD) [Wu2009]_
    is noise-assisted technique, which is meant to be more robust
    than simple Empirical Mode Decomposition (EMD). The robustness is
    checked by performing many decompositions on signals slightly
    perturbed from their initial position. In the grand average over
    all IMF results the noise will cancel each other out and the result
    is pure decomposition.

    Parameters
    ----------
    trials : int (default: 100)
        Number of trials or EMD performance with added noise.
    noise_width : float (default: 0.05)
        Standard deviation of Gaussian noise (:math:`\hat\sigma`).
        It's relative to absolute amplitude of the signal, i.e.
        :math:`\hat\sigma = \sigma\cdot|\max(S)-\min(S)|`, where
        :math:`\sigma` is noise_width.
    ext_EMD : EMD (default: None)
        One can pass EMD object defined outside, which will be
        used to compute IMF decompositions in each trial. If none
        is passed then EMD with default options is used.

    References
    ----------
    .. [Wu2009] Z. Wu and N. E. Huang, "Ensemble empirical mode decomposition:
        A noise-assisted data analysis method", Advances in Adaptive
        Data Analysis, Vol. 1, No. 1 (2009) 1-41.
    """

    logger = logging.getLogger(__name__)

    noise_kinds_all = ["normal", "uniform"]

    def __init__(self, trials=100, noise_width=0.05, ext_EMD=None, **config):

        # Ensemble constants
        self.trials = trials
        self.noise_width = noise_width

        self.random = np.random.RandomState()
        self.noise_kind = "normal"

        if ext_EMD is None:
            from PyEMD import EMD
            self.EMD = EMD()
        else:
            self.EMD = ext_EMD

        # By default (None) Pool spawns #processes = #CPU
        processes = None if "processes" not in config else config["processes"]
        self.pool = Pool(processes=processes)

        # Update based on options
        for key in config.keys():
            if key in self.__dict__.keys():
                self.__dict__[key] = config[key]
            elif key in self.EMD.__dict__.keys():
                self.EMD.__dict__[key] = config[key]

    def __call__(self, S, T=None, max_imf=-1):
        return self.eemd(S, T=T, max_imf=max_imf)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
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

    def eemd(self, S, T=None, max_imf=-1):
        """
        Performs EEMD on provided signal.

        For a large number of iterations defined by `trials` attr
        the method performs :py:meth:`emd` on a signal with added white noise.

        Parameters
        ----------
        S : numpy array,
            Input signal on which EEMD is performed.
        T : numpy array, (default: None)
            If none passed samples are numerated.
        max_imf : int, (default: -1)
            Defines up to how many IMFs each decomposition should
            be performed. By default (negative value) it decomposes
            all IMFs.

        Returns
        -------
        eIMF : numpy array
            Set of ensemble IMFs produced from input signal. In general,
            these do not have to be, and most likely will not be, same as IMFs
            produced using EMD.
        """
        if T is None: T = np.arange(len(S), dtype=S.dtype)

        scale = self.noise_width*np.abs(np.max(S)-np.min(S))
        self._S = S
        self._T = T
        self._N = N = len(S)
        self._scale = scale
        self.max_imf = max_imf

        # For trial number of iterations perform EMD on a signal
        # with added white noise
        all_IMFs = self.pool.map(self._trial_update, range(self.trials))

        max_imfNo = max([IMFs.shape[0] for IMFs in all_IMFs])

        self.E_IMF = np.zeros((max_imfNo, N))
        for IMFs in all_IMFs:
            self.E_IMF[:IMFs.shape[0]] += IMFs

        return self.E_IMF/self.trials

    def _trial_update(self, trial):
        # Generate noise
        noise = self.generate_noise(self._scale, self._N)
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
    global E_imfNo
    E_imfNo = np.zeros(50, dtype=np.int)

    # Logging options
    logging.basicConfig(level=logging.INFO)

    # EEMD options
    max_imf = -1

    # Signal options
    N = 500
    tMin, tMax = 0, 2*np.pi
    T = np.linspace(tMin, tMax, N)

    S = 3*np.sin(4*T) + 4*np.cos(9*T) + np.sin(8.11*T+1.2)

    # Prepare and run EEMD
    eemd = EEMD()
    eemd.trials = 50
    eemd.noise_seed(12345)

    E_IMFs = eemd.eemd(S, T, max_imf)
    imfNo  = E_IMFs.shape[0]

    # Plot results in a grid
    c = np.floor(np.sqrt(imfNo+1))
    r = np.ceil( (imfNo+1)/c)

    plt.ioff()
    plt.subplot(r,c,1)
    plt.plot(T, S, 'r')
    plt.xlim((tMin, tMax))
    plt.title("Original signal")

    for num in range(imfNo):
        plt.subplot(r,c,num+2)
        plt.plot(T, E_IMFs[num],'g')
        plt.xlim((tMin, tMax))
        plt.title("Imf "+str(num+1))

    plt.show()
