#!/usr/bin/python
# Coding: UTF-8

from __future__ import print_function

import numpy as np
from PyEMD import EEMD
import unittest

class ExtremaTest(unittest.TestCase):

    cmp_msg = lambda _,a,b: "Expected {}, Returned {}".format(a,b)

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
        noise_kind = 'uniform'
        spline_kind = 'linear'

        # Making sure that we are not testing default options
        eemd = EEMD()

        self.assertFalse(eemd.trials==trials,
                self.cmp_msg(eemd.trials, trials))

        self.assertFalse(eemd.noise_kind==noise_kind,
                self.cmp_msg(eemd.noise_kind, noise_kind))

        self.assertFalse(eemd.EMD.spline_kind==spline_kind,
                self.cmp_msg(eemd.EMD.spline_kind, spline_kind))

        # Testing for passing attributes via params
        params = {"trials": trials, "noise_kind": noise_kind,
                  "spline_kind": spline_kind}
        eemd = EEMD(**params)

        self.assertTrue(eemd.trials==trials,
                self.cmp_msg(eemd.trials, trials))

        self.assertTrue(eemd.noise_kind==noise_kind,
                self.cmp_msg(eemd.noise_kind, noise_kind))

        self.assertTrue(eemd.EMD.spline_kind==spline_kind,
                self.cmp_msg(eemd.EMD.spline_kind, spline_kind))

    def test_eemd_unsupportedNoiseKind(self):
        noise_kind = "whoever_supports_this_is_wrong"
        eemd = EEMD(noise_kind=noise_kind)

        with self.assertRaises(ValueError):
            eemd.generate_noise(1., 100)

    def test_eemd_passingCustomEMD(self):

        spline_kind = "linear"
        params = {"spline_kind": spline_kind}

        eemd = EEMD()
        self.assertFalse(eemd.EMD.spline_kind==spline_kind,
                "Not"+self.cmp_msg(eemd.EMD.spline_kind, spline_kind))

        from PyEMD import EMD

        emd = EMD(**params)

        eemd = EEMD(ext_EMD=emd)
        self.assertTrue(eemd.EMD.spline_kind==spline_kind,
                self.cmp_msg(eemd.EMD.spline_kind, spline_kind))

if __name__ == "__main__":
    unittest.main()
