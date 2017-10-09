#!/usr/bin/python
# Coding: UTF-8

from __future__ import print_function

import numpy as np
from PyEMD import EMD
import unittest

class EMDTest(unittest.TestCase):

    cmp_msg = lambda _,a,b: "Expected {}, Returned {}".format(a,b)

    def test_default_call_EMD(self):
        T = np.arange(50)
        S = np.cos(T*0.1)
        max_imf = 2

        emd = EMD()
        results = emd(S, T, max_imf)

    def test_different_length_input(self):
        T = np.arange(20)
        S = np.random.random(len(T)+7)

        emd = EMD()
        with self.assertRaises(ValueError):
            emd.emd(S, T)

    def test_trend(self):
        """
        Input is trend. Expeting no shifting process.
        """
        emd = EMD()

        t = np.arange(0, 1, 0.01)
        S = 2*t

        # Input - linear function f(t) = 2*t
        IMF = emd.emd(S, t)
        self.assertEqual(IMF.shape[0], 1, "Expecting single IMF")
        self.assertTrue(np.allclose(S, IMF[0]))

    def test_single_imf(self):
        """
        Input is IMF. Expecint single shifting.
        """

        maxDiff = lambda a,b: np.max(np.abs(a-b))

        emd = EMD()
        emd.FIXE_H = 2

        t = np.arange(0, 1, 0.001)
        c1 = np.cos(4*2*np.pi*t) # 2 Hz
        S = c1.copy()

        # Input - linear function f(t) = sin(2Hz t)
        IMF = emd.emd(S, t)
        self.assertEqual(IMF.shape[0], 1, "Expecting sin + trend")

        diff = np.allclose(IMF[0], c1)
        self.assertTrue(diff, "Expecting 1st IMF to be sin\nMaxDiff = "+str(maxDiff(IMF[0],c1)))

        # Input - linear function f(t) = siin(2Hz t) + 2*t
        c2 = 5*(t+2)
        S += c2.copy()
        IMF = emd.emd(S, t)

        self.assertEqual(IMF.shape[0], 2, "Expecting sin + trend")
        diff1 = np.allclose(IMF[0], c1, atol=0.2)
        self.assertTrue(diff1, "Expecting 1st IMF to be sin\nMaxDiff = "+str(maxDiff(IMF[0],c1)))
        diff2 = np.allclose(IMF[1], c2, atol=0.2)
        self.assertTrue(diff2, "Expecting 2nd IMF to be trend\nMaxDiff = "+str(maxDiff(IMF[1],c2)))

    def test_emd_passArgsViaDict(self):
        FIXE = 10
        params = {"FIXE": FIXE, "nothing": 0}

        # First test without initiation
        emd = EMD()
        self.assertFalse(emd.FIXE==FIXE, "{} == {}".format(emd.FIXE, FIXE))

        # Second: test with passing
        emd = EMD(**params)
        self.assertTrue(emd.FIXE==FIXE, "{} == {}".format(emd.FIXE, FIXE))

    def test_emd_FIXE(self):
        T = np.linspace(0, 1, 100)
        c = np.sin(9*2*np.pi*T)
        offset = 4
        S = c + offset

        emd = EMD()

        # Default state: converge 
        self.assertTrue(emd.FIXE==0)
        self.assertTrue(emd.FIXE_H==0)

        # Set 1 iteration per each sift,
        # same as removing offset
        FIXE = 1
        emd.FIXE = FIXE

        # Check flags correctness
        self.assertTrue(emd.FIXE==FIXE)
        self.assertTrue(emd.FIXE_H==0)

        # Extract IMFs
        IMFs = emd.emd(S)

        # Check that IMFs are correct
        self.assertTrue(np.allclose(IMFs[0], c))
        self.assertTrue(np.allclose(IMFs[1], offset))

    def test_emd_FIXEH(self):
        T = np.linspace(0, 2, 200)
        c1 = 1*np.sin(11*2*np.pi*T+0.1)
        c2 = 11*np.sin(1*2*np.pi*T+0.1)
        offset = 9
        S = c1 + c2 + offset

        emd = EMD()

        # Default state: converge 
        self.assertTrue(emd.FIXE==0)
        self.assertTrue(emd.FIXE_H==0)

        # Set 5 iterations per each protoIMF
        FIXE_H = 6
        emd.FIXE_H = FIXE_H

        # Check flags correctness
        self.assertTrue(emd.FIXE==0)
        self.assertTrue(emd.FIXE_H==FIXE_H)

        # Extract IMFs
        IMFs = emd.emd(S)

        # Check that IMFs are correct
        self.assertTrue(IMFs.shape[0]==3)

        closeIMF1 = np.allclose(c1[2:-2], IMFs[0,2:-2], atol=0.2)
        self.assertTrue(closeIMF1)
        self.assertTrue(np.allclose(c1, IMFs[0], atol=1.))

        closeIMF2 = np.allclose(c2[2:-2], IMFs[1,2:-2], atol=0.21)
        self.assertTrue(closeIMF2)
        self.assertTrue(np.allclose(c2, IMFs[1], atol=1.))

        closeOffset = np.allclose(offset, IMFs[2,2:-2], atol=0.1)
        self.assertTrue(closeOffset)

        closeOffset = np.allclose(offset, IMFs[2,1:-1], atol=0.5)
        self.assertTrue(closeOffset)

    def test_emd_default(self):
        T = np.linspace(0, 2, 200)
        c1 = 1*np.sin(11*2*np.pi*T+0.1)
        c2 = 11*np.sin(1*2*np.pi*T+0.1)
        offset = 9
        S = c1 + c2 + offset

        emd = EMD(spline_kind='akima')
        IMFs = emd.emd(S, T)

        self.assertTrue(IMFs.shape[0]==3)

        closeIMF1 = np.allclose(c1[2:-2], IMFs[0,2:-2], atol=0.2)
        self.assertTrue(closeIMF1)

        closeIMF2 = np.allclose(c2[2:-2], IMFs[1,2:-2], atol=0.21)
        self.assertTrue(closeIMF2)

        closeOffset = np.allclose(offset, IMFs[2,1:-1], atol=0.5)
        self.assertTrue(closeOffset)

if __name__ == "__main__":
    unittest.main()
