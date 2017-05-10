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
import pylab as py

class EEMD:

    logger = logging.getLogger(__name__)

    def __init__(self):

        # Import libraries
        from PyEMD.EMD import EMD

        # Ensemble constants
        self.noiseWidth = 0.3
        self.trials = 100

        self.EMD = EMD()
        self.EMD.FIXE_H = 5

    def eemd(self, S, timeLine=None, maxImf=None):

        if timeLine is None: timeLine = np.arange(len(S), dtype=S.dtype)
        if maxImf is None: maxImf = -1

        N = len(S)
        E_IMF = np.zeros((1,N))

        for trial in range(self.trials):
            self.logger.debug("trial: "+str(trial))

            noise = np.random.normal(loc=0, scale=self.noiseWidth, size=N)

            tmpIMFs = self.emd(S+noise, timeLine, maxImf)
            imfNo = tmpIMFs.shape[0]

            while(E_IMF.shape[0] < imfNo):
                E_IMF = np.vstack((E_IMF, np.zeros(N)))

            E_IMF[:imfNo] += tmpIMFs

        E_IMF /= self.trials

        return E_IMF

    def emd(self, S, timeLine, maxImf=-1):
        return self.EMD.emd(S, timeLine, maxImf)

###################################################
## Beggining of program

if __name__ == "__main__":

    # Logging options
    logging.basicConfig(level=logging.INFO)

    PLOT = 0
    INTERACTIVE = 1

    REPEATS = 1

    N = 500
    maxImf = -1

    t = timeLine = np.linspace(0, 2*np.pi, N)

    n = 2
    s1 = 3*np.sin(4*t)
    s2 = 4*np.sin(9*t)
    s3 = np.sin(11*t)
    S = np.sum( [-eval("s%i"%i) for i in range(1,1+n)], axis=0)

    S = np.random.normal(0,1, len(t))
    IMFs = EEMD().eemd(S, timeLine, maxImf)
    imfNo  = IMFs.shape[0]

    c = np.floor(np.sqrt(imfNo+1))
    r = np.ceil( (imfNo+1)/c)

    py.ioff()
    py.subplot(r,c,1)
    py.plot(timeLine, S, 'r')
    py.title("Original signal")

    for num in range(imfNo):
        py.subplot(r,c,num+2)
        py.plot(timeLine, IMFs[num],'g')
        py.title("Imf no " +str(num) )

    py.show()
