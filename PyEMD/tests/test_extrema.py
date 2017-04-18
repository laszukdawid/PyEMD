#!/usr/bin/python
# Coding: UTF-8

from __future__ import print_function

import numpy as np
from PyEMD.EMD import EMD
import unittest

class ExtremaTest(unittest.TestCase):


    def test_find_extrema_simple(self):
        """
        Simple test for extrema.
        """
        emd = EMD()
        emd.extrema_detection = "simple"

        t = np.arange(10)
        s = np.array([-1, 0, 1, 0, -1, 0, 3, 0, -9, 0])
        expMaxPos = [2, 6]
        expMaxVal = [1, 3]
        expMinPos = [4, 8]
        expMinVal = [-1, -9]

        maxPos, maxVal, minPos, minVal, nz = emd.find_extrema(t, s)

        self.assertEqual(maxPos.tolist(), expMaxPos)
        self.assertEqual(maxVal.tolist(), expMaxVal)
        self.assertEqual(minPos.tolist(), expMinPos)
        self.assertEqual(minVal.tolist(), expMinVal)

    def test_find_extrema_simple_repeat(self):
        """
        Test what happens in /^^\ situation, i.e.
        when extremum is somewhere between two consecutive pts.
        """
        emd = EMD()
        emd.extrema_detection = "simple"

        t = np.arange(2,13)
        s = np.array([-1, 0, 1, 1, 0, -1, 0, 3, 0, -9, 0])
        expMaxPos = [5, 9]
        expMaxVal = [1, 3]
        expMinPos = [7, 11]
        expMinVal = [-1, -9]

        maxPos, maxVal, minPos, minVal, nz = emd.find_extrema(t, s)

        self.assertEqual(maxPos.tolist(), expMaxPos)
        self.assertEqual(maxVal.tolist(), expMaxVal)
        self.assertEqual(minPos.tolist(), expMinPos)
        self.assertEqual(minVal.tolist(), expMinVal)

    def test_bound_extrapolation_simple(self):
        emd = EMD()
        emd.extrema_detection = "simple"
        emd.nbsym = 1
        emd.DTYPE = np.int64

        S = [ 2, 0,-3, 1, 2, 4, 3,-2, 0, 1, 2, 1, 0, 1, 2, 5, 4, 0,-2,-1]
        S = np.array(S)
        T = np.arange(len(S))

        pp = emd.prepare_points

        # There are 4 cases for both (L)eft and (R)ight ends. In case of left (L) bound:
        # L1) ,/ -- ext[0] is min, s[0] < ext[1] (1st max)
        # L2) / -- ext[0] is min, s[0] > ext[1] (1st max)
        # L3) ^. -- ext[0] is max, s[0] > ext[1] (1st min)
        # L4) \ -- ext[0] is max, s[0] < ext[1] (1st min)

        ## CASE 1
        # L1, R1 -- no edge MIN & no edge MIN 
        s = S.copy()
        t = T.copy()

        maxPos, maxVal, minPos, minVal, nz = emd.find_extrema(t, s)

        # Should extrapolate left and right bounds
        maxExtrema, minExtrema = pp(t, s, \
                        maxPos, maxVal, minPos, minVal)

        self.assertEqual([-1,5,10,15,21], maxExtrema[0].tolist())
        self.assertEqual([4,4,2,5,5], maxExtrema[1].tolist())
        self.assertEqual([-3,2,7,12,18,24], minExtrema[0].tolist())
        self.assertEqual([-2,-3,-2,0,-2,0], minExtrema[1].tolist())

        ## CASE 2
        # L2, R2 -- edge MIN, edge MIN
        s = S[2:-1].copy()
        t = T[2:-1].copy()

        maxPos, maxVal, minPos, minVal, nz = emd.find_extrema(t, s)

        # Should extrapolate left and right bounds
        maxExtrema, minExtrema = pp(t, s, \
                        maxPos, maxVal, minPos, minVal)

        self.assertEqual([-1,5,10,15,21], maxExtrema[0].tolist())
        self.assertEqual([4,4,2,5,5], maxExtrema[1].tolist())
        self.assertEqual([-3,2,7,12,18], minExtrema[0].tolist())
        self.assertEqual([-2,-3,-2,0,-2], minExtrema[1].tolist())

        ## CASE 3
        # L3, R3 -- no edge MAX & no edge MAX  
        s, t = S[3:-3], T[3:-3]

        maxPos, maxVal, minPos, minVal, nz = emd.find_extrema(t, s)

        # Should extrapolate left and right bounds
        maxExtrema, minExtrema = pp(t, s, \
                        maxPos, maxVal, minPos, minVal)

        self.assertEqual([0,5,10,15,20], maxExtrema[0].tolist())
        self.assertEqual([2,4,2,5,2], maxExtrema[1].tolist())
        self.assertEqual([3,7,12,18], minExtrema[0].tolist())
        self.assertEqual([-2,-2,0,0], minExtrema[1].tolist())

        ## CASE 4
        # L4, R4 -- edge MAX & edge MAX
        s, t = S[5:-4], T[5:-4]

        maxPos, maxVal, minPos, minVal, nz = emd.find_extrema(t, s)

        # Should extrapolate left and right bounds
        maxExtrema, minExtrema = pp(t, s, \
                        maxPos, maxVal, minPos, minVal)

        self.assertEqual([0,5,10,15], maxExtrema[0].tolist())
        self.assertEqual([2,4,2,5], maxExtrema[1].tolist())
        self.assertEqual([3,7,12,18], minExtrema[0].tolist())
        self.assertEqual([-2,-2,0,0], minExtrema[1].tolist())

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

        maxPos, maxVal, minPos, minVal, nz = emd.find_extrema(t, s)

        self.assertEqual(maxPos.tolist(), expMaxPos)
        self.assertEqual(maxVal.tolist(), expMaxVal)
        self.assertEqual(minPos.tolist(), expMinPos)
        self.assertEqual(minVal.tolist(), expMinVal)

    def test_find_extrema_parabol_repeat(self):
        """
        Test what happens in /^^\ situation, i.e.
        when extremum is somewhere between two consecutive pts.
        """
        emd = EMD()
        emd.extrema_detection = "parabol"

        t = np.arange(2,13)
        s = np.array([-1, 0, 1, 1, 0, -1, 0, 3, 0, -9, 0])
        expMaxPos = [4.5, 9]
        expMaxVal = [1.25, 3]
        expMinPos = [7, 11]
        expMinVal = [-1, -9]

        maxPos, maxVal, minPos, minVal, nz = emd.find_extrema(t, s)

        self.assertEqual(maxPos.tolist(), expMaxPos)
        self.assertEqual(maxVal.tolist(), expMaxVal)
        self.assertEqual(minPos.tolist(), expMinPos)
        self.assertEqual(minVal.tolist(), expMinVal)

    # TODO:
    #   - nbsym > 1
    #   - case when extremum doesn't go outside edge, mirror in ref to the sig's init

if __name__ == "__main__":
    unittest.main()
