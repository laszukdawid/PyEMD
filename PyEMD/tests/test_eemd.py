#!/usr/bin/python
# Coding: UTF-8

from __future__ import print_function

import numpy as np
from PyEMD import EEMD
import unittest

class ExtremaTest(unittest.TestCase):

    def test_eemd_simpleRun(self):
        T = np.linspace(0, 1, 100)
        S = np.sin(2*np.pi*T)

        eemd = EEMD(trials=10, max_imf=1)
        eemd.EMD.FIXE_H = 5
        eemd.eemd(S)

    def test_eemd_passingArgumentsViaDict(self):
        T = np.linspace(0, 1, 100)
        S = np.sin(2*np.pi*T)

        trials = 10
        spline_kind = 'linear'

        params = {"trials": trials, "spline_kind": spline_kind}
        eemd = EEMD(**params)

        cmp_msg = lambda a,b: "Expected {}, Returned {}".format(a,b)
        self.assertTrue(eemd.trials==trials,
                cmp_msg(eemd.trials, trials))

        self.assertTrue(eemd.EMD.spline_kind==spline_kind,
                cmp_msg(eemd.EMD.spline_kind, spline_kind))

    def test_eemd_passingCustomEMD(self):

        spline_kind = "linear"
        params = {"spline_kind": spline_kind}

        eemd = EEMD()
        self.assertFalse(eemd.EMD.spline_kind==spline_kind)

        from PyEMD import EMD

        emd = EMD(**params)

        eemd = EEMD(ext_EMD=emd)
        self.assertTrue(eemd.EMD.spline_kind==spline_kind)

if __name__ == "__main__":
    unittest.main()
