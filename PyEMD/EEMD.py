#!/usr/bin/python
# coding: UTF-8
#
# Author:   Dawid Laszuk
# Contact:  https://github.com/laszukdawid/PyEMD/issues
#
# Feel free to contact for any information.
"""
.. currentmodule:: EEMD
"""

from __future__ import print_function

import logging
import numpy as np

from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, Optional, Sequence, Tuple, Union

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
    parallel : bool (default: False)
        Flag whether to use multiprocessing in EEMD execution.
        Since each EMD(s+noise) is independent this should improve execution
        speed considerably.
        *Note* that it's disabled by default because it's the most common
        problem when EEMD takes too long time to finish.
        If you set the flag to True, make also sure to set `processes` to
        some reasonable value.

    Keyword parameters
    ------------------
    processes : int or None (optional)
        Number of processes harness when executing in parallel mode.
        The value should be between 1 and max that depends on your hardware.
    separate_trends : bool (default: False)
        Flag whether to isolate trends from each EMD decosition into a separate component.
        If `true`, the resulting EEMD will contain ensemble only from IMFs and
        the mean residue will be stacked as the last element.

    References
    ----------
    .. [Wu2009] Z. Wu and N. E. Huang, "Ensemble empirical mode decomposition:
        A noise-assisted data analysis method", Advances in Adaptive
        Data Analysis, Vol. 1, No. 1 (2009) 1-41.
    """

    logger = logging.getLogger(__name__)

    noise_kinds_all = ["normal", "uniform"]

    def __init__(self, trials: int=100, noise_width: float=0.05, ext_EMD=None, parallel: bool=False, **kwargs):

        # Ensemble constants
        self.trials = trials
        self.noise_width = noise_width
        self.separate_trends = bool(kwargs.get('separate_trends', False))

        self.random = np.random.RandomState()
        self.noise_kind = kwargs.get('noise_kind', 'normal')
        self.parallel = parallel
        self.processes = kwargs.get('processes')  # Optional[int]
        if self.processes is not None and not self.parallel:
            self.logger.warning("Passed value for process has no effect when `parallel` is False.")

        if ext_EMD is None:
            from PyEMD import EMD
            self.EMD = EMD(**kwargs)
        else:
            self.EMD = ext_EMD

        self.E_IMF = None  # Optional[np.ndarray]
        self.residue = None  # Optional[np.ndarray]

    def __call__(self, S: np.ndarray, T: Optional[np.ndarray]=None, max_imf: int=-1) -> np.ndarray:
        return self.eemd(S, T=T, max_imf=max_imf)

    def __getstate__(self) -> Dict:
        self_dict = self.__dict__.copy()
        if 'pool' in self_dict:
            del self_dict['pool']
        return self_dict

    def generate_noise(self, scale: float, size: Union[int, Sequence[int]]) -> np.ndarray:
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

    def noise_seed(self, seed: int) -> None:
        """Set seed for noise generation."""
        self.random.seed(seed)

    def eemd(self, S: np.ndarray, T: Optional[np.ndarray]=None, max_imf: int=-1) -> np.ndarray:
        """
        Performs EEMD on provided signal.

        For a large number of iterations defined by `trials` attr
        the method performs :py:meth:`emd` on a signal with added white noise.

        Parameters
        ----------
        S : numpy array,
            Input signal on which EEMD is performed.
        T : numpy array or None, (default: None)
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
        if self.parallel:
            pool = Pool(processes=self.processes)

            all_IMFs = pool.map(self._trial_update, range(self.trials))

            pool.close()

        else:  # Not parallel
            all_IMFs = map(self._trial_update, range(self.trials))

        proto_eimf = defaultdict(lambda: np.zeros(N))
        proto_count = defaultdict(int)
        for (imfs, trend) in all_IMFs:

            # A bit of explanation here.
            # If the `trend` is not None, that means it was intentionally separated in the decomp process.
            # This might due to `separate_trends` flag which means that trends are summed up separately
            # and treated as the last component. Since `proto_eimfs` is a dict, that `-1` is treated literally
            # and **not** as the *last position*. We can then use that `-1` to always add it as the last pos
            # in the actual eIMF, which indicates the trend.
            if trend is not None:
                proto_eimf[-1] += trend
                proto_count[-1] += 1

            for imf_num, imf in enumerate(imfs):
                proto_eimf[imf_num] += imf
                proto_count[imf_num] += 1

        self.E_IMF = np.zeros((len(proto_eimf), N))

        # *Note*: We are diving by proto-imf count per order, rather than blantly by num of trails.
        #         This means we're calculating a mean over all observations rather.
        for imf_num, imf in proto_eimf.items():
            self.E_IMF[imf_num] += imf / proto_count[imf_num]

        self.residue = S - np.sum(self.E_IMF, axis=0)

        return self.E_IMF

    def _trial_update(self, trial) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """A single trial evaluation, i.e. EMD(signal + noise).

        *Note*: Although `trial` argument isn't used it's needed for the (multiprocessing) map method.
        """
        noise = self.generate_noise(self._scale, self._N)
        imfs = self.emd(self._S+noise, self._T, self.max_imf)
        trend = None
        if self.separate_trends:
            imfs, trend = self.EMD.get_imfs_and_trend()
        
        return (imfs, trend)

    def emd(self, S: np.ndarray, T: np.ndarray, max_imf: int=-1) -> np.ndarray:
        """Vanilla EMD method.

        Provides emd evaluation from provided EMD class.
        For reference please see :class:`PyEMD.EMD`.
        """
        return self.EMD.emd(S, T, max_imf)

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and residue from recently analysed signal.

        Returns
        -------
        (imfs, residue) : (np.ndarray, np.ndarray)
            Tuple that contains all imfs and a residue (if any).

        """
        if self.E_IMF is None or self.residue is None:
            raise ValueError('No IMF found. Please, run EMD method or its variant first.')
        return self.E_IMF, self.residue


###################################################
## Beginning of program

if __name__ == "__main__":

    import pylab as plt
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
    eemd = EEMD(trials=50)
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
