import unittest

import numpy as np

from PyEMD import EMD
from PyEMD.visualisation import Visualisation


class VisTest(unittest.TestCase):
    def test_instantiation(self):
        emd = EMD()
        with self.assertRaises(ValueError):
            Visualisation(emd)

    def test_instantiation2(self):
        t = np.linspace(0, 1, 50)
        S = t + np.cos(np.cos(4.0 * t ** 2))
        emd = EMD()
        emd.emd(S, t)
        imfs, res = emd.get_imfs_and_residue()
        vis = Visualisation(emd)
        self.assertTrue(np.alltrue(vis.imfs == imfs))
        self.assertTrue(np.alltrue(vis.residue == res))

    def test_check_imfs(self):
        vis = Visualisation()
        imfs = np.arange(50).reshape(2, 25)
        res = np.arange(25)
        imfs, res = vis._check_imfs(imfs, res, False)
        self.assertEqual(len(imfs), 2)

    def test_check_imfs2(self):
        vis = Visualisation()
        with self.assertRaises(AttributeError):
            vis._check_imfs(None, None, False)

    def test_check_imfs3(self):
        vis = Visualisation()
        imfs = np.arange(50).reshape(2, 25)

        out_imfs, out_res = vis._check_imfs(imfs, None, False)

        self.assertTrue(np.alltrue(imfs == out_imfs))
        self.assertIsNone(out_res)

    def test_check_imfs4(self):
        vis = Visualisation()
        imfs = np.arange(50).reshape(2, 25)
        with self.assertRaises(AttributeError):
            vis._check_imfs(imfs, None, True)

    def test_check_imfs5(self):
        t = np.linspace(0, 1, 50)
        S = t + np.cos(np.cos(4.0 * t ** 2))
        emd = EMD()
        emd.emd(S, t)
        imfs, res = emd.get_imfs_and_residue()
        vis = Visualisation(emd)
        imfs2, res2 = vis._check_imfs(imfs, res, False)
        self.assertTrue(np.alltrue(imfs == imfs2))
        self.assertTrue(np.alltrue(res == res2))

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
        S = t + np.cos(np.cos(4.0 * t ** 2))
        emd = EMD()
        imfs = emd.emd(S, t)
        vis = Visualisation()
        phase = vis._calc_inst_phase(imfs, 0.4)
        assert len(imfs) == len(phase)

    def test_calc_instant_phase3(self):
        t = np.linspace(0, 1, 50)
        S = t + np.cos(np.cos(4.0 * t ** 2))
        emd = EMD()
        imfs = emd.emd(S, t)
        vis = Visualisation()
        with self.assertRaises(AssertionError):
            _ = vis._calc_inst_phase(imfs, 0.8)

    def test_calc_instant_freq_alphaNone(self):
        t = np.linspace(0, 1, 50)
        S = t + np.cos(np.cos(4.0 * t ** 2))
        emd = EMD()
        imfs = emd.emd(S, t)
        vis = Visualisation()
        freqs = vis._calc_inst_freq(imfs, t, False, None)
        self.assertEqual(imfs.shape, freqs.shape)

    def test_calc_instant_freq(self):
        t = np.linspace(0, 1, 50)
        S = t + np.cos(np.cos(4.0 * t ** 2))
        emd = EMD()
        imfs = emd.emd(S, t)
        vis = Visualisation()
        freqs = vis._calc_inst_freq(imfs, t, False, 0.4)
        self.assertEqual(imfs.shape, freqs.shape)

    def test_plot_instant_freq(self):
        vis = Visualisation()
        t = np.arange(20)
        with self.assertRaises(AttributeError):
            vis.plot_instant_freq(t)
