#!/usr/bin/python
# Coding: UTF-8

import unittest

import numpy as np

from PyEMD import EMD
from PyEMD.splines import *


class IMFTest(unittest.TestCase):
    """
    Since these tests depend heavily on NumPy & SciPy,
    make sure you have NumPy >= 1.12 and SciPy >= 0.19.
    """

    def test_unsupporter_spline(self):
        emd = EMD()
        emd.spline_kind = "waterfall"

        S = np.random.random(20)

        with self.assertRaises(ValueError):
            emd.emd(S)

    def test_akima(self):
        dtype = np.float32

        emd = EMD()
        emd.spline_kind = "akima"
        emd.DTYPE = dtype

        # Test error: len(X)!=len(Y)
        with self.assertRaises(ValueError):
            akima(np.array([0]), np.array([1, 2]), np.array([0, 1, 2]))

        # Test error: any(dt) <= 0
        with self.assertRaises(ValueError):
            akima(np.array([1, 0, 2]), np.array([1, 2]), np.array([0, 1, 2]))
        with self.assertRaises(ValueError):
            akima(np.array([0, 0, 2]), np.array([1, 2]), np.array([0, 1, 1]))

        # Test for correct responses
        T = np.array([0, 1, 2, 3, 4], dtype)
        S = np.array([0, 1, -1, -1, 5], dtype)
        t = np.array([i / 2.0 for i in range(9)], dtype)

        _t, s = emd.spline_points(t, np.array((T, S)))
        s_true = np.array([S[0], 0.9125, S[1], 0.066666, S[2], -1.35416667, S[3], 1.0625, S[4]], dtype)

        self.assertTrue(np.allclose(s_true, s), "Comparing akima with true")

        s_np = akima(np.array(T), np.array(S), np.array(t))
        self.assertTrue(np.allclose(s, s_np), "Shouldn't matter if with numpy")

    def test_cubic(self):
        dtype = np.float64

        emd = EMD()
        emd.spline_kind = "cubic"
        emd.DTYPE = dtype

        T = np.array([0, 1, 2, 3, 4], dtype=dtype)
        S = np.array([0, 1, -1, -1, 5], dtype=dtype)
        t = np.arange(9, dtype=dtype) / 2.0

        # TODO: Something weird with float32.
        # Seems to be SciPy problem.
        _t, s = emd.spline_points(t, np.array((T, S)))

        s_true = np.array([S[0], 1.203125, S[1], 0.046875, S[2], -1.515625, S[3], 1.015625, S[4]], dtype=dtype)
        self.assertTrue(np.allclose(s, s_true, atol=0.01), "Comparing cubic")

        T = T[:-2].copy()
        S = S[:-2].copy()
        t = np.arange(5, dtype=dtype) / 2.0

        _t, s3 = emd.spline_points(t, np.array((T, S)))

        s3_true = np.array([S[0], 0.78125, S[1], 0.28125, S[2]], dtype=dtype)
        self.assertTrue(np.allclose(s3, s3_true), "Compare cubic 3pts")

    def test_slinear(self):
        dtype = np.float64

        emd = EMD()
        emd.spline_kind = "slinear"
        emd.DTYPE = dtype

        T = np.array([0, 1, 2, 3, 4], dtype=dtype)
        S = np.array([0, 1, -1, -1, 5], dtype=dtype)
        t = np.arange(9, dtype=dtype) / 2.0

        _t, s = emd.spline_points(t, np.array((T, S)))

        s_true = np.array([S[0], 0.5, S[1], 0, S[2], -1, S[3], 2, S[4]], dtype=dtype)
        self.assertTrue(np.allclose(s, s_true), "Comparing SLinear")


if __name__ == "__main__":
    unittest.main()
