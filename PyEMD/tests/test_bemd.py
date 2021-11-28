#!/usr/bin/python
# Coding: UTF-8
import unittest

import numpy as np

try:
    from PyEMD.BEMD import BEMD
except (ImportError, ModuleNotFoundError):
    # Not supported until supported.
    pass


@unittest.skip("Not supported until supported")
class BEMDTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bemd = BEMD()

    @staticmethod
    def _generate_image(r=64, c=64):
        return np.random.random((r, c))

    @staticmethod
    def _generate_linear_image(r=16, c=16):
        rows = np.arange(r)
        return np.repeat(rows, c).reshape(r, c)

    @staticmethod
    def _generate_Gauss(x, y, pos, std, amp=1):
        x_s = x - pos[0]
        y_s = y - pos[1]
        x2 = x_s * x_s
        y2 = y_s * y_s
        exp = np.exp(-(x2 + y2) / (2 * std * std))
        # exp[exp<1e-6] = 0
        scale = amp / np.linalg.norm(exp)
        return scale * exp

    @classmethod
    def _sin(cls, x_n=128, y_n=128, x_f=[1], y_f=[0], dx=0, dy=0):
        x = np.linspace(0, 1, x_n) - dx
        y = np.linspace(0, 1, y_n) - dy
        xv, yv = np.meshgrid(x, y)
        img = np.zeros(xv.shape)
        for f in x_f:
            img += np.sin(f * 2 * np.pi * xv)
        for f in y_f:
            img += np.cos(f * 2 * np.pi * yv)
        return 255 * (img - img.min()) / (img.max() - img.min())

    def test_extract_maxima(self):
        image = self._sin(x_n=32, y_n=32, y_f=[1], dy=1)
        max_peak_x, max_peak_y = BEMD.extract_maxima_positions(image)
        self.assertEqual(max_peak_x.size, 6)  # Clustering
        self.assertEqual(max_peak_y.size, 6)  # Clustering

    def test_extract_minima(self):
        image = self._sin(x_n=64, y_n=64, y_f=[2])
        min_peak_x, min_peak_y = BEMD.extract_minima_positions(image)
        self.assertEqual(min_peak_x.size, 16)  # Clustering
        self.assertEqual(min_peak_y.size, 16)  # Clustering

    def test_find_extrema(self):
        image = self._sin()
        min_peaks, max_peaks = BEMD.find_extrema_positions(image)
        self.assertTrue(isinstance(min_peaks, tuple))
        self.assertTrue(isinstance(min_peaks[0], np.ndarray))
        self.assertTrue(isinstance(max_peaks[1], np.ndarray))

    def test_default_call_BEMD(self):
        x = np.arange(50)
        y = np.arange(50)
        xv, yv = np.meshgrid(x, y)
        img = self._generate_Gauss(xv, yv, (10, 20), 5)
        img += self._sin(x_n=x.size, y_n=y.size)

        max_imf = 2
        self.bemd(img, max_imf)

    def test_endCondition_perfectReconstruction(self):
        c1 = self._generate_image()
        c2 = self._generate_image()
        IMFs = np.stack((c1, c2))
        org_img = np.sum(IMFs, axis=0)
        self.assertTrue(self.bemd.end_condition(org_img, IMFs))

    def test_bemd_simpleIMF(self):
        image = self._sin(x_f=[3, 5], y_f=[2])
        IMFs = self.bemd(image)

        # One of the reasons this algorithm isn't the preferred one
        self.assertTrue(IMFs.shape[0] == 7, "Depending on spline, there should be an IMF and possibly trend")

    def test_bemd_limitImfNo(self):

        # Create image
        rows, cols = 64, 64
        linear_background = 0.2 * self._generate_linear_image(rows, cols)

        # Sinusoidal IMF
        X = np.arange(cols)[None, :].T
        Y = np.arange(rows)
        x_comp_1d = np.sin(X * 0.3) + np.cos(X * 2.9) ** 2
        y_comp_1d = np.sin(Y * 0.2)
        comp_2d = 10 * x_comp_1d * y_comp_1d
        comp_2d = comp_2d

        image = linear_background + comp_2d

        # Limit number of IMFs
        max_imf = 2

        # decompose image
        IMFs = self.bemd(image, max_imf=max_imf)

        # It should have no more than 2 (max_imf) + residue
        self.assertEqual(IMFs.shape[0], 1 + max_imf)


if __name__ == "__main__":
    unittest.main()
