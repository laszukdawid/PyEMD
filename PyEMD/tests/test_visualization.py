#!/usr/bin/python
# Coding: UTF-8

from __future__ import print_function

import numpy as np
from PyEMD import EMD, EEMD, CEEMDAN
from PyEMD.visualisation import Visualisation
import unittest

class VisTest(unittest.TestCase):

    def test_instantiation(self):
        emd = EMD()
        with self.assertRaises(ValueError):
            Visualisation(emd)

    def test_instantiation2(self):
        t = np.linspace(0, 1, 50)
        S = t + np.cos(np.cos(4.*t**2))
        emd = EMD()
        emd.emd(S, t)
        imfs, res = emd.get_imfs_and_residue()
        vis = Visualisation(emd)
        assert (vis.imfs == imfs).all()
        assert (vis.residue == res).all()

    def test_check_imfs(self):
        vis = Visualisation()
        imfs = np.arange(50).reshape(2,25)
        res = np.arange(25)
        imfs, res = vis._check_imfs(imfs, res, False)
        assert len(imfs) == 2

    def test_check_imfs2(self):
        vis = Visualisation()
        with self.assertRaises(AttributeError):
            vis._check_imfs(None, None, False)

    def test_check_imfs3(self):
        vis = Visualisation()
        imfs = np.arange(50).reshape(2,25)
        vis._check_imfs(imfs, None, False)

    def test_check_imfs4(self):
        vis = Visualisation()
        imfs = np.arange(50).reshape(2,25)
        with self.assertRaises(AttributeError):
            vis._check_imfs(imfs, None, True)

    def test_check_imfs5(self):
        t = np.linspace(0, 1, 50)
        S = t + np.cos(np.cos(4.*t**2))
        emd = EMD()
        emd.emd(S, t)
        imfs, res = emd.get_imfs_and_residue()
        vis = Visualisation(emd)
        imfs2, res2 = vis._check_imfs(imfs, res, False)
        assert (imfs == imfs2).all()
        assert (res == res2).all()

    def test_plot_imfs(self):
        vis = Visualisation()
        with self.assertRaises(AttributeError):
            vis.plot_imfs()

    # Does not work for Python 2.7 (TravisCI), even with Agg backend
    # def test_plot_imfs2(self):
    #     t = np.linspace(0, 1, 50)
    #     S = t + np.cos(np.cos(4.*t**2))
    #     emd = EMD()
    #     emd.emd(S, t)
    #     vis = Visualisation(emd)
    #     vis.plot_imfs()

    def test_calc_instant_phase(self):
        sig = np.arange(10)
        vis = Visualisation()
        phase = vis._calc_inst_phase(sig, None)
        assert len(sig) == len(phase)

    def test_calc_instant_phase2(self):
        t = np.linspace(0, 1, 50)
        S = t + np.cos(np.cos(4.*t**2))
        emd = EMD()
        imfs = emd.emd(S, t)
        vis = Visualisation()
        phase = vis._calc_inst_phase(imfs, 0.4)
        assert len(imfs) == len(phase)

    def test_calc_instant_phase3(self):
        t = np.linspace(0, 1, 50)
        S = t + np.cos(np.cos(4.*t**2))
        emd = EMD()
        imfs = emd.emd(S, t)
        vis = Visualisation()
        with self.assertRaises(AssertionError):
            phase = vis._calc_inst_phase(imfs, 0.8)

    def test_calc_instant_freq(self):
        t = np.linspace(0, 1, 50)
        S = t + np.cos(np.cos(4.*t**2))
        emd = EMD()
        imfs = emd.emd(S, t)
        vis = Visualisation()
        freqs = vis._calc_inst_freq(imfs, t, False, None)
        assert imfs.shape == freqs.shape

    def test_calc_instant_freq(self):
        t = np.linspace(0, 1, 50)
        S = t + np.cos(np.cos(4.*t**2))
        emd = EMD()
        imfs = emd.emd(S, t)
        vis = Visualisation()
        freqs = vis._calc_inst_freq(imfs, t, False, 0.4)
        assert imfs.shape == freqs.shape

    def test_plot_instant_freq(self):
        vis = Visualisation()
        t = np.arange(20)
        with self.assertRaises(AttributeError):
            vis.plot_instant_freq(t)


if __name__ == "__main__":
    unittest.main()
