#!/usr/bin/python
# Coding: UTF-8

from __future__ import print_function

import numpy as np
from PyEMD.EMD import EMD
import unittest

class IMFTest(unittest.TestCase):

    def compare_signals(self, a, b, delta=None):
        if delta==None:
            delta = 1e-13
        return np.all(np.abs(a-b)<delta)

    def test_trend(self):
        """
        Input is trend. Expeting no shifting process.
        """
        emd = EMD()

        t = np.arange(0, 1, 0.01)
        S = 2*t

        # Input - linear function f(t) = 2*t
        IMF, EXT, ITER, imfNo = emd.emd(S, t)
        self.assertEqual(len(IMF.keys()), 1, "Expecting single IMF")

    def test_single_imf(self):
        """
        Input is IMF. Expecint single shifting.
        """

        maxDiff = lambda a,b: np.max(np.abs(a-b))

        emd = EMD()
        emd.FIXE_H = 5

        t = np.arange(0, 1, 0.001)
        c1 = np.cos(4*2*np.pi*t) # 2 Hz
        S = c1.copy()

        # Input - linear function f(t) = sin(2Hz t)
        IMF, EXT, ITER, imfNo = emd.emd(S, t)
        self.assertEqual(len(IMF.keys()), 1, "Expecting sin + trend")

        diff = self.compare_signals(IMF[0], c1, delta=1e-10)
        self.assertTrue(diff, "Expecting 1st IMF to be sin\nMaxDiff = "+str(maxDiff(IMF[0],c1)))

        # Input - linear function f(t) = siin(2Hz t) + 2*t
        c2 = 5*(t+2)
        S += c2.copy()
        IMF, EXT, ITER, imfNo = emd.emd(S, t)

        self.assertEqual(len(IMF.keys()), 2, "Expecting sin + trend")
        diff1 = self.compare_signals(IMF[0], c1, delta=0.2)
        self.assertTrue(diff1, "Expecting 1st IMF to be sin\nMaxDiff = "+str(maxDiff(IMF[0],c1)))
        diff2 = self.compare_signals(IMF[1], c2, delta=0.2)
        self.assertTrue(diff2, "Expecting 2nd IMF to be trend\nMaxDiff = "+str(maxDiff(IMF[1],c2)))

if __name__ == "__main__":
    unittest.main()
