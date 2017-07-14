#!/usr/bin/python
# Coding: UTF-8

from __future__ import print_function

import numpy as np
from PyEMD import EMD
import unittest

class EMDTest(unittest.TestCase):

    cmp_msg = lambda _,a,b: "Expected {}, Returned {}".format(a,b)

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

        closeIMF1 = np.allclose(c1[1:-1], IMFs[0,1:-1], atol=0.5)
        self.assertTrue(closeIMF1)
        self.assertTrue(np.allclose(c1, IMFs[0], atol=1.))

        closeIMF2 = np.allclose(c2[1:-1], IMFs[1,1:-1], atol=0.5)
        self.assertTrue(closeIMF2)
        self.assertTrue(np.allclose(c2, IMFs[1], atol=1.))

        closeOffset = np.allclose(offset, IMFs[2,1:-1], atol=0.5)
        self.assertTrue(closeOffset)
        self.assertTrue(np.allclose(IMFs[1,1:-1], c2[1:-1], atol=0.5))

        closeOffset = np.allclose(offset, IMFs[2,1:-1], atol=0.5)
        self.assertTrue(closeOffset)

    def test_emd_default(self):
        T = np.linspace(0, 2, 200)
        c1 = 1*np.sin(11*2*np.pi*T+0.1)
        c2 = 11*np.sin(1*2*np.pi*T+0.1)
        offset = 9
        S = c1 + c2 + offset

        emd = EMD()
        IMFs = emd.emd(S, T)

        self.assertTrue(IMFs.shape[0]==3)

        closeIMF1 = np.allclose(c1[1:-1], IMFs[0,1:-1], atol=0.5)
        self.assertTrue(closeIMF1)

        closeIMF2 = np.allclose(c2[1:-1], IMFs[1,1:-1], atol=0.5)
        self.assertTrue(closeIMF2)

        closeOffset = np.allclose(offset, IMFs[2,1:-1], atol=0.5)
        self.assertTrue(closeOffset)

if __name__ == "__main__":
    unittest.main()
