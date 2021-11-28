import unittest

import numpy as np

from PyEMD import EMD


class EMDTest(unittest.TestCase):
    @staticmethod
    def test_default_call_EMD():
        T = np.arange(50)
        S = np.cos(T * 0.1)
        max_imf = 2

        emd = EMD()
        emd(S, T, max_imf)

    def test_different_length_input(self):
        T = np.arange(20)
        S = np.random.random(len(T) + 7)

        emd = EMD()
        with self.assertRaises(ValueError):
            emd.emd(S, T)

    def test_trend(self):
        """
        Input is trend. Expeting no shifting process.
        """
        emd = EMD()

        t = np.arange(0, 1, 0.01)
        S = 2 * t

        # Input - linear function f(t) = 2*t
        imfs = emd.emd(S, t)
        self.assertEqual(imfs.shape[0], 1, "Expecting single IMF")
        self.assertTrue(np.allclose(S, imfs[0]))

    def test_single_imf(self):
        """
        Input is IMF. Expecint single shifting.
        """

        def max_diff(a, b):
            return np.max(np.abs(a - b))

        emd = EMD()
        emd.FIXE_H = 2

        t = np.arange(0, 1, 0.001)
        c1 = np.cos(4 * 2 * np.pi * t)  # 2 Hz
        S = c1.copy()

        # Input - linear function f(t) = sin(2Hz t)
        imfs = emd.emd(S, t)
        self.assertEqual(imfs.shape[0], 1, "Expecting sin + trend")

        diff = np.allclose(imfs[0], c1)
        self.assertTrue(diff, "Expecting 1st IMF to be sin\nMaxDiff = " + str(max_diff(imfs[0], c1)))

        # Input - linear function f(t) = siin(2Hz t) + 2*t
        c2 = 5 * (t + 2)
        S += c2.copy()
        imfs = emd.emd(S, t)

        self.assertEqual(imfs.shape[0], 2, "Expecting sin + trend")
        diff1 = np.allclose(imfs[0], c1, atol=0.2)
        self.assertTrue(diff1, "Expecting 1st IMF to be sin\nMaxDiff = " + str(max_diff(imfs[0], c1)))
        diff2 = np.allclose(imfs[1], c2, atol=0.2)
        self.assertTrue(diff2, "Expecting 2nd IMF to be trend\nMaxDiff = " + str(max_diff(imfs[1], c2)))

    def test_emd_passArgsViaDict(self):
        FIXE = 10
        params = {"FIXE": FIXE, "nothing": 0}

        # First test without initiation
        emd = EMD()
        self.assertFalse(emd.FIXE == FIXE, "{} == {}".format(emd.FIXE, FIXE))

        # Second: test with passing
        emd = EMD(**params)
        self.assertTrue(emd.FIXE == FIXE, "{} == {}".format(emd.FIXE, FIXE))

    def test_emd_passImplicitParamsDirectly(self):
        FIXE = 10
        svar_thr = 0.2

        # First test without initiation
        emd = EMD()
        self.assertFalse(emd.FIXE == FIXE, "{} == {}".format(emd.FIXE, FIXE))

        # Second: test with passing
        emd = EMD(FIXE=FIXE, svar_thr=svar_thr, nothing=0)
        self.assertTrue(emd.FIXE == FIXE, "{} == {}".format(emd.FIXE, FIXE))
        self.assertTrue(emd.svar_thr == svar_thr, "{} == {}".format(emd.svar_thr, svar_thr))

    def test_emd_FIXE(self):
        T = np.linspace(0, 1, 100)
        c = np.sin(9 * 2 * np.pi * T)
        offset = 4
        S = c + offset

        emd = EMD()

        # Default state: converge
        self.assertTrue(emd.FIXE == 0)
        self.assertTrue(emd.FIXE_H == 0)

        # Set 1 iteration per each sift,
        # same as removing offset
        FIXE = 1
        emd.FIXE = FIXE

        # Check flags correctness
        self.assertTrue(emd.FIXE == FIXE)
        self.assertTrue(emd.FIXE_H == 0)

        # Extract IMFs
        IMFs = emd.emd(S)

        # Check that IMFs are correct
        self.assertTrue(np.allclose(IMFs[0], c))
        self.assertTrue(np.allclose(IMFs[1], offset))

    def test_emd_FIXEH(self):
        T = np.linspace(0, 2, 200)
        c1 = 1 * np.sin(11 * 2 * np.pi * T + 0.1)
        c2 = 11 * np.sin(1 * 2 * np.pi * T + 0.1)
        offset = 9
        S = c1 + c2 + offset

        emd = EMD()

        # Default state: converge
        self.assertTrue(emd.FIXE == 0)
        self.assertTrue(emd.FIXE_H == 0)

        # Set 5 iterations per each protoIMF
        FIXE_H = 6
        emd.FIXE_H = FIXE_H

        # Check flags correctness
        self.assertTrue(emd.FIXE == 0)
        self.assertTrue(emd.FIXE_H == FIXE_H)

        # Extract IMFs
        imfs = emd.emd(S)

        # Check that IMFs are correct
        self.assertTrue(imfs.shape[0] == 3)

        close_imf1 = np.allclose(c1[2:-2], imfs[0, 2:-2], atol=0.2)
        self.assertTrue(close_imf1)
        self.assertTrue(np.allclose(c1, imfs[0], atol=1.0))

        close_imf2 = np.allclose(c2[2:-2], imfs[1, 2:-2], atol=0.21)
        self.assertTrue(close_imf2)
        self.assertTrue(np.allclose(c2, imfs[1], atol=1.0))

        close_offset = np.allclose(offset, imfs[2, 2:-2], atol=0.1)
        self.assertTrue(close_offset)

        close_offset = np.allclose(offset, imfs[2, 1:-1], atol=0.5)
        self.assertTrue(close_offset)

    def test_emd_default(self):
        T = np.linspace(0, 2, 200)
        c1 = 1 * np.sin(11 * 2 * np.pi * T + 0.1)
        c2 = 11 * np.sin(1 * 2 * np.pi * T + 0.1)
        offset = 9
        S = c1 + c2 + offset

        emd = EMD(spline_kind="akima")
        imfs = emd.emd(S, T)
        self.assertTrue(imfs.shape[0] == 3)

        close_imfs1 = np.allclose(c1[2:-2], imfs[0, 2:-2], atol=0.21)
        self.assertTrue(close_imfs1)

        close_imfs2 = np.allclose(c2[2:-2], imfs[1, 2:-2], atol=0.24)
        self.assertTrue(close_imfs2)

        close_offset = np.allclose(offset, imfs[2, 1:-1], atol=0.5)
        self.assertTrue(close_offset)

    def test_max_iteration_flag(self):
        S = np.random.random(200)
        emd = EMD()
        emd.MAX_ITERATION = 10
        emd.FIXE = 20

        imfs = emd.emd(S)

        # There's not much to test, except that it doesn't fail.
        # With low MAX_ITERATION value for random signal it's
        # guaranteed to have at least 2 imfs.
        self.assertTrue(imfs.shape[0] > 1)

    def test_get_imfs_and_residue(self):
        S = np.random.random(200)
        emd = EMD(**{"MAX_ITERATION": 10, "FIXE": 20})
        all_imfs = emd(S, max_imf=3)

        imfs, residue = emd.get_imfs_and_residue()
        self.assertEqual(all_imfs.shape[0], imfs.shape[0] + 1, "Compare number of components")
        self.assertTrue(np.array_equal(all_imfs[:-1], imfs), "Shouldn't matter where imfs are from")
        self.assertTrue(np.array_equal(all_imfs[-1], residue), "Residue, if any, is the last row")

    def test_get_imfs_and_residue_without_running(self):
        emd = EMD()
        with self.assertRaises(ValueError):
            _, _ = emd.get_imfs_and_residue()

    def test_get_imfs_and_trend(self):
        emd = EMD()
        T = np.linspace(0, 2 * np.pi, 100)
        expected_trend = 5 * T
        S = 2 * np.sin(4.1 * 6.28 * T) + 1.2 * np.cos(7.4 * 6.28 * T) + expected_trend

        all_imfs = emd(S)
        imfs, trend = emd.get_imfs_and_trend()

        onset_trend = trend - trend.mean()
        onset_expected_trend = expected_trend - expected_trend.mean()
        self.assertEqual(all_imfs.shape[0], imfs.shape[0] + 1, "Compare number of components")
        self.assertTrue(np.array_equal(all_imfs[:-1], imfs), "Shouldn't matter where imfs are from")
        self.assertTrue(
            np.allclose(onset_trend, onset_expected_trend, rtol=0.1, atol=0.5),
            "Extracted trend should be close to the actual trend",
        )


if __name__ == "__main__":
    unittest.main()
