#!/usr/bin/python
# coding: UTF-8
#
# Author:   Dawid Laszuk
# Contact:  laszukdawid@gmail.com
#
# Feel free to contact for any information.

from __future__ import print_function

import logging
import numpy as np

class EEMD:

    logger = logging.getLogger(__name__)

    def __init__(self):

        # Import libraries
        from PyEMD import EMD

        # Ensemble constants
        self.noise_width = 0.3
        self.trials = 100

        self.EMD = EMD()
        self.EMD.FIXE_H = 5

    def eemd(self, S, T=None, max_imf=None):

        if T is None: T = np.arange(len(S), dtype=S.dtype)
        if max_imf is None: max_imf = -1

        N = len(S)
        E_IMF = np.zeros((1,N))

        for trial in range(self.trials):
            self.logger.debug("trial: "+str(trial))

            noise = np.random.normal(loc=0, scale=self.noise_width, size=N)

            IMFs = self.emd(S+noise, T, max_imf)
            imfNo = IMFs.shape[0]

            while(E_IMF.shape[0] < imfNo):
                E_IMF = np.vstack((E_IMF, np.zeros(N)))

            E_IMF[:imfNo] += IMFs

        E_IMF /= self.trials

        return E_IMF

    def emd(self, S, T, max_imf=-1):
        return self.EMD.emd(S, T, max_imf)

###################################################
## Beginning of program

if __name__ == "__main__":

    import pylab as plt

    # Logging options
    logging.basicConfig(level=logging.INFO)

    # EEMD options
    PLOT = 0
    INTERACTIVE = 1
    REPEATS = 1

    max_imf = -1

    # Signal options
    N = 500
    tMin, tMax = 0, 2*np.pi
    T = np.linspace(tMin, tMax, N)

    S = 3*np.sin(4*T) + 4*np.cos(9*T) + np.sin(8.11*T+1.2)

    # Prepare and run EEMD 
    eemd = EEMD()
    eemd.trials = 50

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
