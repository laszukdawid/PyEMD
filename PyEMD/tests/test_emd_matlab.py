import unittest

import numpy as np

from PyEMD.EMD_matlab import EMD


class EMDMatlabTest(unittest.TestCase):
    @staticmethod
    def test_default_call_EMD():
        T = np.arange(0, 1, 0.01)
        S = np.cos(2 * T * 2 * np.pi)
        max_imf = 2

        emd = EMD()
        emd.emd(emd, S, T, max_imf)

    def test_different_length_input(self):
        T = np.arange(20)
        S = np.random.random(len(T) + 7)

        emd = EMD()
        with self.assertRaises(ValueError):
            emd.emd(emd, S, T)

    def test_trend(self):
        """
        Input is trend. Expeting no shifting process.
        """
        emd = EMD()

        T = np.arange(0, 1, 0.01)
        S = np.cos(2 * T * 2 * np.pi)

        # Input - linear function f(t) = 2*t
        output = emd.emd(emd, S, T)
        self.assertEqual(len(output), 4, "Expecting 4 outputs - IMF, EXT, ITER, imfNo")

        IMF, EXT, ITER, imfNo = output
        self.assertEqual(len(IMF), 2, "Expecting single IMF + residue")
        self.assertEqual(len(IMF[0]), len(S), "Expecting single IMF")
        self.assertTrue(np.allclose(S, IMF[0]))
        self.assertLessEqual(ITER[0], 5, "Expecting 5 iterations at most")
        self.assertEqual(imfNo, 2, "Expecting 1 IMF")
        self.assertEqual(EXT[0], 3, "Expecting single EXT")


if __name__ == "__main__":
    unittest.main()
