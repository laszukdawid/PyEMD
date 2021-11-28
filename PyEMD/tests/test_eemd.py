import unittest

import numpy as np

from PyEMD import EEMD


class EEMDTest(unittest.TestCase):
    @staticmethod
    def cmp_msg(a, b):
        return "Expected {}, Returned {}".format(a, b)

    @staticmethod
    def test_default_call_EEMD():
        T = np.arange(50)
        S = np.cos(T * 0.1)
        max_imf = 2

        eemd = EEMD()
        eemd(S, T, max_imf)

    def test_eemd_simpleRun(self):
        T = np.linspace(0, 1, 100)
        S = np.sin(2 * np.pi * T)

        config = {"processes": 1}
        eemd = EEMD(trials=10, max_imf=1, **config)
        eemd.EMD.FIXE_H = 5
        eemd.eemd(S)

        self.assertTrue("processes" in eemd.__dict__)
        self.assertTrue(eemd.processes == 1)

    def test_eemd_passingArgumentsViaDict(self):
        trials = 10
        noise_kind = "uniform"
        spline_kind = "linear"

        # Making sure that we are not testing default options
        eemd = EEMD()

        self.assertFalse(eemd.trials == trials, self.cmp_msg(eemd.trials, trials))

        self.assertFalse(eemd.noise_kind == noise_kind, self.cmp_msg(eemd.noise_kind, noise_kind))

        self.assertFalse(eemd.EMD.spline_kind == spline_kind, self.cmp_msg(eemd.EMD.spline_kind, spline_kind))

        # Testing for passing attributes via params
        params = {"trials": trials, "noise_kind": noise_kind, "spline_kind": spline_kind}
        eemd = EEMD(**params)

        self.assertTrue(eemd.trials == trials, self.cmp_msg(eemd.trials, trials))

        self.assertTrue(eemd.noise_kind == noise_kind, self.cmp_msg(eemd.noise_kind, noise_kind))

        self.assertTrue(eemd.EMD.spline_kind == spline_kind, self.cmp_msg(eemd.EMD.spline_kind, spline_kind))

    def test_eemd_passingArgumentsDirectly(self):
        trials = 10
        noise_kind = "uniform"
        spline_kind = "linear"

        # Making sure that we are not testing default options
        eemd = EEMD()

        self.assertFalse(eemd.trials == trials, self.cmp_msg(eemd.trials, trials))

        self.assertFalse(eemd.noise_kind == noise_kind, self.cmp_msg(eemd.noise_kind, noise_kind))

        self.assertFalse(eemd.EMD.spline_kind == spline_kind, self.cmp_msg(eemd.EMD.spline_kind, spline_kind))

        # Testing for passing attributes via params
        eemd = EEMD(trials=trials, noise_kind=noise_kind, spline_kind=spline_kind)

        self.assertTrue(eemd.trials == trials, self.cmp_msg(eemd.trials, trials))

        self.assertTrue(eemd.noise_kind == noise_kind, self.cmp_msg(eemd.noise_kind, noise_kind))

        self.assertTrue(eemd.EMD.spline_kind == spline_kind, self.cmp_msg(eemd.EMD.spline_kind, spline_kind))

    def test_eemd_unsupportedNoiseKind(self):
        noise_kind = "whoever_supports_this_is_wrong"
        eemd = EEMD(noise_kind=noise_kind)

        with self.assertRaises(ValueError):
            eemd.generate_noise(1.0, 100)

    def test_eemd_passingCustomEMD(self):

        spline_kind = "linear"
        params = {"spline_kind": spline_kind}

        eemd = EEMD()
        self.assertFalse(eemd.EMD.spline_kind == spline_kind, "Not" + self.cmp_msg(eemd.EMD.spline_kind, spline_kind))

        from PyEMD import EMD

        emd = EMD(**params)

        eemd = EEMD(ext_EMD=emd)
        self.assertTrue(eemd.EMD.spline_kind == spline_kind, self.cmp_msg(eemd.EMD.spline_kind, spline_kind))

    def test_eemd_noiseSeed(self):
        T = np.linspace(0, 1, 100)
        S = np.sin(2 * np.pi * T + 4 ** T) + np.cos((T - 0.4) ** 2)

        # Compare up to machine epsilon
        def cmpMachEps(x, y):
            return np.abs(x - y) <= 2 * np.finfo(x.dtype).eps

        config = {"processes": 1}
        eemd = EEMD(trials=10, **config)

        # First run random seed
        eIMF1 = eemd(S)

        # Second run with defined seed, diff than first
        eemd.noise_seed(12345)
        eIMF2 = eemd(S)

        # Extremly unlikely to have same seed, thus different results
        msg_false = "Different seeds, expected different outcomes"
        if eIMF1.shape == eIMF2.shape:
            self.assertFalse(np.all(cmpMachEps(eIMF1, eIMF2)), msg_false)

        # Third run with same seed as with 2nd
        eemd.noise_seed(12345)
        eIMF3 = eemd(S)

        # Using same seeds, thus expecting same results
        msg_true = "Used same seed, expected same results"
        self.assertTrue(np.all(cmpMachEps(eIMF2, eIMF3)), msg_true)

    def test_eemd_notParallel(self):
        S = np.random.random(100)

        eemd = EEMD(trials=5, max_imf=2, parallel=False)
        eemd.EMD.FIXE_H = 2
        eIMFs = eemd.eemd(S)

        self.assertTrue(eIMFs.shape[0] > 0)
        self.assertTrue(eIMFs.shape[1], len(S))
        self.assertFalse("pool" in eemd.__dict__)

    def test_imfs_and_residue_accessor(self):
        S = np.random.random(100)
        eemd = EEMD(trials=5, max_imf=2, parallel=False)
        eIMFs = eemd(S)

        imfs, residue = eemd.get_imfs_and_residue()
        self.assertEqual(eIMFs.shape[0], imfs.shape[0], "Compare number of components")
        self.assertEqual(len(residue), 100, "Check if residue exists")

    def test_imfs_and_residue_accessor2(self):
        eemd = EEMD()
        with self.assertRaises(ValueError):
            imfs, residue = eemd.get_imfs_and_residue()

    def test_separate_trends(self):
        T = np.linspace(0, 2 * np.pi, 100)
        S = np.sin(T) + 3 * np.sin(3 * T + 0.1) + 0.2 * (T + 0.5) * (T - 2)
        eemd = EEMD(trials=20, separate_trends=True)

        eIMFs = eemd(S)
        for imf in eIMFs[:-1]:
            self.assertLess(abs(imf.mean()), 0.5)
        self.assertGreaterEqual(eIMFs[-1].mean(), 1)

    def test_eemd_ensemble_stats(self):
        T = np.linspace(0, 2 * np.pi, 100)
        S = np.sin(T) + 3 * np.sin(3 * T + 0.1) + 0.2 * (T + 0.5) * (T - 2)
        eemd = EEMD(trials=20, separate_trends=True)

        eIMFs = eemd(S)
        self.assertEqual(type(eemd.all_imfs), dict, "All imfs are stored as a dict")
        self.assertTrue(np.all(eIMFs == eemd.ensemble_mean()), "eIMFs are the mean over ensemble")
        self.assertEqual(eemd.ensemble_count(), [len(imfs) for imfs in eemd.all_imfs.values()])
        self.assertEqual(type(eemd.ensemble_std()), np.ndarray, "Ensemble std exists and it's a numpy array")


if __name__ == "__main__":
    unittest.main()
