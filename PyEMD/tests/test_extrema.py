#!/usr/bin/python
# Coding: UTF-8

import unittest

import numpy as np

from PyEMD import EMD


class ExtremaTest(unittest.TestCase):
    def test_incorrectExtremaDetectionSetup(self):
        extrema_detection = "bubble_gum"

        # Sanity check
        emd = EMD()
        self.assertFalse(emd.extrema_detection == extrema_detection)

        # Assign incorrect extrema_detection
        emd.extrema_detection = extrema_detection
        self.assertTrue(emd.extrema_detection == extrema_detection)

        T = np.arange(10)
        S = np.sin(T)
        max_pos, max_val = np.random.random((2, 3))
        min_pos, min_val = np.random.random((2, 3))

        # Check for Exception
        with self.assertRaises(ValueError):
            emd.prepare_points(T, S, max_pos, max_val, min_pos, min_val)

    def test_wrong_extrema_detection_type(self):
        emd = EMD()
        emd.extrema_detection = "very_complicated"

        t = np.arange(10)
        s = np.array([-1, 0, 1, 0, -1, 0, 3, 0, -9, 0])

        with self.assertRaises(ValueError):
            emd.find_extrema(t, s)

    def test_find_extrema_simple(self):
        """Simple test for extrema."""
        emd = EMD()
        emd.extrema_detection = "simple"

        t = np.arange(10)
        s = np.array([-1, 0, 1, 0, -1, 0, 3, 0, -9, 0])
        expMaxPos = [2, 6]
        expMaxVal = [1, 3]
        expMinPos = [4, 8]
        expMinVal = [-1, -9]
        expZeros = t[s == 0]

        maxPos, maxVal, minPos, minVal, nz = emd.find_extrema(t, s)

        self.assertEqual(maxPos.tolist(), expMaxPos)
        self.assertEqual(maxVal.tolist(), expMaxVal)
        self.assertEqual(minPos.tolist(), expMinPos)
        self.assertEqual(minVal.tolist(), expMinVal)
        self.assertEqual(nz.tolist(), expZeros.tolist())

    def test_find_extrema_simple_repeat(self):
        r"""
        Test what happens in /^^\ situation, i.e.
        when extremum is somewhere between two consecutive pts.
        """
        emd = EMD()
        emd.extrema_detection = "simple"

        t = np.arange(2, 13)
        s = np.array([-1, 0, 1, 1, 0, -1, 0, 3, 0, -9, 0])
        expMaxPos = [4, 9]
        expMaxVal = [1, 3]
        expMinPos = [7, 11]
        expMinVal = [-1, -9]

        maxPos, maxVal, minPos, minVal, _ = emd.find_extrema(t, s)

        self.assertEqual(maxPos.tolist(), expMaxPos)
        self.assertEqual(maxVal.tolist(), expMaxVal)
        self.assertEqual(minPos.tolist(), expMinPos)
        self.assertEqual(minVal.tolist(), expMinVal)

    def test_bound_extrapolation_simple(self):
        emd = EMD()
        emd.extrema_detection = "simple"
        emd.nbsym = 1
        emd.DTYPE = np.int64

        S = [0, -3, 1, 4, 3, 2, -2, 0, 1, 2, 1, 0, 1, 2, 5, 4, 0, -2, -1]
        S = np.array(S)
        T = np.arange(len(S))

        pp = emd.prepare_points

        # There are 4 cases for both (L)eft and (R)ight ends. In case of left (L) bound:
        # L1) ,/ -- ext[0] is min, s[0] < ext[1] (1st max)
        # L2) / -- ext[0] is min, s[0] > ext[1] (1st max)
        # L3) ^. -- ext[0] is max, s[0] > ext[1] (1st min)
        # L4) \ -- ext[0] is max, s[0] < ext[1] (1st min)

        # CASE 1
        # L1, R1 -- no edge MIN & no edge MIN
        s = S.copy()
        t = T.copy()

        maxPos, maxVal, minPos, minVal, _ = emd.find_extrema(t, s)

        # Should extrapolate left and right bounds
        maxExtrema, minExtrema = pp(t, s, maxPos, maxVal, minPos, minVal)

        self.assertEqual([-1, 3, 9, 14, 20], maxExtrema[0].tolist())
        self.assertEqual([4, 4, 2, 5, 5], maxExtrema[1].tolist())
        self.assertEqual([-4, 1, 6, 11, 17, 23], minExtrema[0].tolist())
        self.assertEqual([-2, -3, -2, 0, -2, 0], minExtrema[1].tolist())

        # CASE 2
        # L2, R2 -- edge MIN, edge MIN
        s = S[1:-1].copy()
        t = np.arange(s.size)

        maxPos, maxVal, minPos, minVal, _ = emd.find_extrema(t, s)

        # Should extrapolate left and right bounds
        maxExtrema, minExtrema = pp(t, s, maxPos, maxVal, minPos, minVal)

        self.assertEqual([-2, 2, 8, 13, 19], maxExtrema[0].tolist())
        self.assertEqual([4, 4, 2, 5, 5], maxExtrema[1].tolist())
        self.assertEqual([0, 5, 10, 16], minExtrema[0].tolist())
        self.assertEqual([-3, -2, 0, -2], minExtrema[1].tolist())

        # CASE 3
        # L3, R3 -- no edge MAX & no edge MAX
        s = S[2:-3].copy()
        t = np.arange(s.size)

        maxPos, maxVal, minPos, minVal, _ = emd.find_extrema(t, s)

        # Should extrapolate left and right bounds
        maxExtrema, minExtrema = pp(t, s, maxPos, maxVal, minPos, minVal)

        self.assertEqual([-5, 1, 7, 12, 17], maxExtrema[0].tolist())
        self.assertEqual([2, 4, 2, 5, 2], maxExtrema[1].tolist())
        self.assertEqual([-2, 4, 9, 15], minExtrema[0].tolist())
        self.assertEqual([-2, -2, 0, 0], minExtrema[1].tolist())

        # CASE 4
        # L4, R4 -- edge MAX & edge MAX
        s = S[3:-4].copy()
        t = np.arange(s.size)

        maxPos, maxVal, minPos, minVal, _ = emd.find_extrema(t, s)

        # Should extrapolate left and right bounds
        maxExtrema, minExtrema = pp(t, s, maxPos, maxVal, minPos, minVal)

        self.assertEqual([0, 6, 11], maxExtrema[0].tolist())
        self.assertEqual([4, 2, 5], maxExtrema[1].tolist())
        self.assertEqual([-3, 3, 8, 14], minExtrema[0].tolist())
        self.assertEqual([-2, -2, 0, 0], minExtrema[1].tolist())

    def test_find_extrema_parabol(self):
        """
        Simple test for extrema.
        """
        emd = EMD()
        emd.extrema_detection = "parabol"

        t = np.arange(10)
        s = np.array([-1, 0, 1, 0, -1, 0, 3, 0, -9, 0])
        expMaxPos = [2, 6]
        expMaxVal = [1, 3]
        expMinPos = [4, 8]
        expMinVal = [-1, -9]

        maxPos, maxVal, minPos, minVal, _ = emd.find_extrema(t, s)

        self.assertEqual(maxPos.tolist(), expMaxPos)
        self.assertEqual(maxVal.tolist(), expMaxVal)
        self.assertEqual(minPos.tolist(), expMinPos)
        self.assertEqual(minVal.tolist(), expMinVal)

    def test_find_extrema_parabol_repeat(self):
        r"""
        Test what happens in /^^\ situation, i.e.
        when extremum is somewhere between two consecutive pts.
        """
        emd = EMD()
        emd.extrema_detection = "parabol"

        t = np.arange(2, 13)
        s = np.array([-1, 0, 1, 1, 0, -1, 0, 3, 0, -9, 0])
        expMaxPos = [4.5, 9]
        expMaxVal = [1.125, 3]
        expMinPos = [7, 11]
        expMinVal = [-1, -9]

        maxPos, maxVal, minPos, minVal, _ = emd.find_extrema(t, s)

        self.assertEqual(maxPos.tolist(), expMaxPos)
        self.assertEqual(maxVal.tolist(), expMaxVal)
        self.assertEqual(minPos.tolist(), expMinPos)
        self.assertEqual(minVal.tolist(), expMinVal)

    def test_bound_extrapolation_parabol(self):
        emd = EMD()
        emd.extrema_detection = "parabol"
        emd.nbsym = 1
        emd.DTYPE = np.float64

        S = [0, -3, 1, 4, 3, 2, -2, 0, 1, 2, 1, 0, 1, 2, 5, 4, 0, -2, -1]
        S = np.array(S)
        T = np.arange(len(S))

        pp = emd.prepare_points

        # There are 4 cases for both (L)eft and (R)ight ends. In case of left (L) bound:
        # L1) ,/ -- ext[0] is min, s[0] < ext[1] (1st max)
        # L2) / -- ext[0] is min, s[0] > ext[1] (1st max)
        # L3) ^. -- ext[0] is max, s[0] > ext[1] (1st min)
        # L4) \ -- ext[0] is max, s[0] < ext[1] (1st min)

        # CASE 1
        # L1, R1 -- no edge MIN & no edge MIN
        s = S.copy()
        t = T.copy()

        maxPos, maxVal, minPos, minVal, _ = emd.find_extrema(t, s)

        # Should extrapolate left and right bounds
        maxExtrema, minExtrema = pp(t, s, maxPos, maxVal, minPos, minVal)

        maxExtrema = np.round(maxExtrema, decimals=3)
        minExtrema = np.round(minExtrema, decimals=3)

        self.assertEqual([-1.393, 3.25, 9, 14.25, 20.083], maxExtrema[0].tolist())
        self.assertEqual([4.125, 4.125, 2, 5.125, 5.125], maxExtrema[1].tolist())
        self.assertEqual([-4.31, 0.929, 6.167, 11, 17.167, 23.333], minExtrema[0].tolist())
        self.assertEqual([-2.083, -3.018, -2.083, 0, -2.042, 0], minExtrema[1].tolist())

        # CASE 2
        # L2, R2 -- edge MIN, edge MIN
        s = S[1:-1].copy()
        t = T[1:-1].copy()

        maxPos, maxVal, minPos, minVal, _ = emd.find_extrema(t, s)

        # Should extrapolate left and right bounds
        maxExtrema, minExtrema = pp(t, s, maxPos, maxVal, minPos, minVal)

        maxExtrema = np.round(maxExtrema, decimals=3)
        minExtrema = np.round(minExtrema, decimals=3)

        self.assertEqual([-1.25, 3.25, 9, 14.25, 19.75], maxExtrema[0].tolist())
        self.assertEqual([4.125, 4.125, 2, 5.125, 5.125], maxExtrema[1].tolist())
        self.assertEqual([1, 6.167, 11, 17], minExtrema[0].tolist())
        self.assertEqual([-3, -2.083, 0, -2], minExtrema[1].tolist())

        # CASE 3
        # L3, R3 -- no edge MAX & no edge MAX
        s = S[2:-3].copy()
        t = T[2:-3].copy()

        maxPos, maxVal, minPos, minVal, _ = emd.find_extrema(t, s)

        # Should extrapolate left and right bounds
        maxExtrema, minExtrema = pp(t, s, maxPos, maxVal, minPos, minVal)

        maxExtrema = np.round(maxExtrema, decimals=3)
        minExtrema = np.round(minExtrema, decimals=3)

        self.assertEqual([-2.5, 3.25, 9, 14.25, 19.5], maxExtrema[0].tolist())
        self.assertEqual([2, 4.125, 2, 5.125, 2], maxExtrema[1].tolist())
        self.assertEqual([0.333, 6.167, 11, 17.5], minExtrema[0].tolist())
        self.assertEqual([-2.083, -2.083, 0, 0], minExtrema[1].tolist())

        # CASE 4
        # L4, R4 -- edge MAX & edge MAX
        s = S[3:-4].copy()
        t = T[3:-4].copy()

        maxPos, maxVal, minPos, minVal, _ = emd.find_extrema(t, s)

        # Should extrapolate left and right bounds
        maxExtrema, minExtrema = pp(t, s, maxPos, maxVal, minPos, minVal)

        maxExtrema = np.round(maxExtrema, decimals=3)
        minExtrema = np.round(minExtrema, decimals=3)

        self.assertEqual([3, 9, 14], maxExtrema[0].tolist())
        self.assertEqual([4, 2, 5], maxExtrema[1].tolist())
        self.assertEqual([-0.167, 6.167, 11, 17], minExtrema[0].tolist())
        self.assertEqual([-2.083, -2.083, 0, 0], minExtrema[1].tolist())

    # TODO:
    #   - nbsym > 1


if __name__ == "__main__":
    unittest.main()
