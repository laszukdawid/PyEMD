#!/usr/bin/python
# Coding: UTF-8

import unittest

import numpy as np
import scipy as sp

from PyEMD.compact import *


class CompactTest(unittest.TestCase):
    @staticmethod
    def create_signal():
        # Do NOT modify this function! If you do, the following can happen:
        # -- Possible test errors for filter (tolerace of np.allclose).
        # -- Error in the derivative test (analytical derivative is hardcoded)
        t = np.linspace(0.0, np.pi, 200)
        return 0.1 * np.cos(2.0 * np.pi * t)

    def test_TDMA(self):

        diags = np.array([0.5 * np.ones(10), 1.0 * np.ones(10), 0.5 * np.ones(10)])
        positions = [-1, 0, 1]
        tridiag = sp.sparse.spdiags(diags, positions, 10, 10).todense()

        # change some diagonal values to make sure it is working
        diags[0][3] = 2.0
        diags[1][1] = 2.0
        tridiag[3, 2] = 2.0
        tridiag[1, 1] = 2.0

        rhs = np.arange(10)

        answer = np.linalg.solve(tridiag, rhs)
        # result = TDMAsolver(*diags, rhs)
        result = TDMAsolver(diags[0], diags[1], diags[2], rhs)

        self.assertTrue(np.allclose(answer, result))

    def test_filter_off(self):
        S = self.create_signal()
        self.assertTrue(np.allclose(S, filt6(S, 0.5), atol=1e-5))

    def test_filter_small(self):
        S = self.create_signal()
        self.assertTrue(np.allclose(S, filt6(S, 0.45), atol=1e-5))

    def test_filter_medium(self):
        S = self.create_signal()
        self.assertTrue(np.allclose(S, filt6(S, 0.0), atol=1e-5))

    def test_filter_high(self):
        S = self.create_signal()
        self.assertTrue(np.allclose(S, filt6(S, -0.5), atol=1e-5))

    def test_pade6(self):
        t = np.linspace(0.0, np.pi, 200)
        S = self.create_signal()
        Sprime = -0.1 * 2.0 * np.pi * np.sin(2.0 * np.pi * t)
        self.assertTrue(np.allclose(Sprime, pade6(S, t[1] - t[0]), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
