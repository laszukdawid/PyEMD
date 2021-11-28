#!/usr/bin/python
# Coding: UTF-8

import unittest

import numpy as np

from PyEMD.EMD2d import EMD2D


@unittest.skip("Not supported until supported")
class ImageEMDTest(unittest.TestCase):
    def setUp(self) -> None:
        self.emd2d = EMD2D()

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

    def test_default_call_EMD2d(self):
        x = np.arange(50)
        y = np.arange(50)
        xv, yv = np.meshgrid(x, y)
        pos = (10, 20)
        std = 5
        img = self._generate_Gauss(xv, yv, pos, std)

        max_imf = 2

        emd2d = EMD2D()
        emd2d(img, max_imf)

    def test_endCondition_perfectReconstruction(self):
        c1 = self._generate_image()
        c2 = self._generate_image()
        IMFs = np.stack((c1, c2))
        org_img = np.sum(IMFs, axis=0)
        self.assertTrue(self.emd2d.end_condition(org_img, IMFs))

    def test_findExtrema_singleMax(self):
        x = np.arange(50)
        y = np.arange(50)
        xv, yv = np.meshgrid(x, y)
        pos = (10, 20)
        std = 5
        img_max = self._generate_Gauss(xv, yv, pos, std)

        idx_min, idx_max = self.emd2d.find_extrema(img_max)
        x_min, y_min = xv[idx_min], yv[idx_min]
        x_max, y_max = xv[idx_max], yv[idx_max]

        self.assertTrue((x_max, y_max) == pos)
        self.assertTrue(len(x_min) == 0)
        self.assertTrue(len(y_min) == 0)

    def test_findExtrema_singleMin(self):
        x = np.arange(50)
        y = np.arange(50)
        xv, yv = np.meshgrid(x, y)
        pos = (10, 20)
        std = 5
        img_max = (-1) * self._generate_Gauss(xv, yv, pos, std, 10)

        idx_min, idx_max = self.emd2d.find_extrema(img_max)
        x_min, y_min = xv[idx_min], yv[idx_min]
        x_max, y_max = xv[idx_max], yv[idx_max]

        self.assertTrue((x_min, y_min) == pos)
        self.assertTrue(len(x_max) == 0)
        self.assertTrue(len(y_max) == 0)

    def test_findExtrema_general(self):
        x = np.arange(50)
        y = np.arange(50)
        xv, yv = np.meshgrid(x, y)

        min_peaks = [((5, 40), 4, -1), ((34, 10), 2, -3)]
        max_peaks = [((10, 20), 5, 2), ((25, 25), 3, 1), ((40, 5), 3, 3)]

        # Construct image with few Gausses
        img = np.zeros(xv.shape)
        for peak in max_peaks + min_peaks:
            img = img + self._generate_Gauss(xv, yv, peak[0], peak[1], peak[2])

        # Extract extrema
        idx_min, idx_max = self.emd2d.find_extrema(img)
        x_min = xv[idx_min].tolist()
        y_min = yv[idx_min].tolist()
        x_max = xv[idx_max].tolist()
        y_max = yv[idx_max].tolist()

        # Confirm that all peaks found - number
        self.assertTrue(len(x_min) == len(min_peaks))
        self.assertTrue(len(y_min) == len(min_peaks))
        self.assertTrue(len(x_max) == len(max_peaks))
        self.assertTrue(len(y_max) == len(max_peaks))

        for peak in min_peaks:
            peak_pos = peak[0]
            x_min.remove(peak_pos[0])
            y_min.remove(peak_pos[1])

        for peak in max_peaks:
            peak_pos = peak[0]
            x_max.remove(peak_pos[0])
            y_max.remove(peak_pos[1])

        # Confirm that all peaks found - exact position
        self.assertTrue(len(x_min) == 0)
        self.assertTrue(len(y_min) == 0)
        self.assertTrue(len(x_max) == 0)
        self.assertTrue(len(y_max) == 0)

    def test_splinePoints_SBS_simpleGrid(self):
        # Test points - leave space inbetween for interpolation
        X = np.arange(5) * 2
        Y = np.arange(5) * 2
        xm, ym = np.meshgrid(X, Y)

        xmf = xm.flatten()
        ymf = ym.flatten()

        # Constant value image
        zf = np.ones(xmf.size)

        # Interpolation grid
        xi = np.arange(10)
        yi = np.arange(10)

        # interpolated_image == np.ones((5,5))
        interpolated_image = self.emd2d.spline_points(xmf, ymf, zf, xi, yi)

        # It is expected, that interpolation on constant will produce
        # constant image
        self.assertTrue(np.allclose(interpolated_image, 1))

    def test_splinePoints_SBS_linearGrid(self):
        X = np.arange(5) * 2
        Y = np.arange(5) * 2
        xm, ym = np.meshgrid(X, Y)

        xmf = xm.flatten()
        ymf = ym.flatten()

        # Linear value image
        z = np.repeat(np.arange(0, 10, 2)[None, :], 5, axis=0)
        zf = z.flatten()

        # Interpolation grid
        xi = np.arange(9)
        yi = np.arange(9)

        # interpolated_image[row] == row
        interpolated_image = self.emd2d.spline_points(xmf, ymf, zf, xi, yi)

        # Since interpolation is on linear function, each row should have
        # value equal to position, i.e. z[0]=(0...), ,,,, z[n] = (n...n).
        for n in range(xi.size):
            nth_row = interpolated_image[n]
            self.assertTrue(np.allclose(nth_row, n))

    def test_emd2d_noExtrema(self):
        linear_image = self._generate_linear_image()

        IMFs = self.emd2d.emd(linear_image)

        self.assertTrue(np.all(linear_image == IMFs))

    def test_emd2d_simpleIMF(self):
        rows, cols = 128, 128

        # Sinusoidal IMF
        X = np.arange(cols)[None, :].T
        Y = np.arange(rows)
        sin_1d = np.sin(Y * 0.3)
        cos_1d = np.cos(X * 0.4)
        comp_2d = 10 * cos_1d * sin_1d
        comp_2d -= np.mean(comp_2d)

        image = comp_2d
        IMFs = self.emd2d.emd(image)

        # Image = IMF + noise
        self.assertTrue(IMFs.shape[0] <= 2, "Depending on spline, there should be an IMF and possibly trend")

        self.assertTrue(
            np.allclose(IMFs[0], image, atol=0.5), "Output: \n" + str(IMFs[0]) + "\nInput: \n" + str(comp_2d)
        )

    def test_emd2d_linearBackground_simpleIMF(self):
        rows, cols = 128, 128
        linear_background = 0.1 * self._generate_linear_image(rows, cols)

        # Sinusoidal IMF
        X = np.arange(cols)[None, :].T
        Y = np.arange(rows)
        x_comp_1d = np.sin(X * 0.5)
        y_comp_1d = np.sin(Y * 0.2)
        comp_2d = 5 * x_comp_1d * y_comp_1d

        image = linear_background + comp_2d
        IMFs = self.emd2d.emd(image)

        # Check that only two IMFs were extracted
        self.assertTrue(IMFs.shape == (2, rows, cols), "Shape is " + str(IMFs.shape))

        # First IMF should be sin
        self.assertTrue(
            np.allclose(IMFs[0], comp_2d, atol=1.0), "Output: \n" + str(IMFs[0]) + "\nInput: \n" + str(comp_2d)
        )

        # Second IMF should be linear trend
        self.assertTrue(np.allclose(IMFs[1], linear_background, atol=1.0))

    def test_emd2d_linearBackground_simpleIMF_FIXE(self):
        rows, cols = 128, 128
        linear_background = 0.1 * self._generate_linear_image(rows, cols)

        # Sinusoidal IMF
        X = np.arange(cols)[None, :].T
        Y = np.arange(rows)
        x_comp_1d = np.sin(X * 0.5)
        y_comp_1d = np.sin(Y * 0.2)
        comp_2d = 5 * x_comp_1d * y_comp_1d

        image = linear_background + comp_2d
        emd2d = EMD2D()
        emd2d.FIXE = 10
        IMFs = emd2d.emd(image)

        # Check that only two IMFs were extracted
        self.assertTrue(IMFs.shape == (2, rows, cols), "Shape is " + str(IMFs.shape))

        # First IMF should be sin
        self.assertTrue(
            np.allclose(IMFs[0], comp_2d, atol=1.0), "Output: \n" + str(IMFs[0]) + "\nInput: \n" + str(comp_2d)
        )

        # Second IMF should be linear trend
        self.assertTrue(np.allclose(IMFs[1], linear_background, atol=1.0))

    def test_emd2d_linearBackground_simpleIMF_FIXE_H(self):
        rows, cols = 128, 128
        linear_background = 0.1 * self._generate_linear_image(rows, cols)

        # Sinusoidal IMF
        X = np.arange(cols)[None, :].T
        Y = np.arange(rows)
        x_comp_1d = np.sin(X * 0.5)
        y_comp_1d = np.sin(Y * 0.2)
        comp_2d = 5 * x_comp_1d * y_comp_1d

        image = linear_background + comp_2d
        emd2D = EMD2D()
        emd2D.FIXE_H = 10
        IMFs = emd2D.emd(image)

        # Check that only two IMFs were extracted
        self.assertTrue(IMFs.shape == (2, rows, cols), "Shape is " + str(IMFs.shape))

        # First IMF should be sin
        self.assertTrue(
            np.allclose(IMFs[0], comp_2d, atol=1.0), "Output: \n" + str(IMFs[0]) + "\nInput: \n" + str(comp_2d)
        )

        # Second IMF should be linear trend
        self.assertTrue(np.allclose(IMFs[1], linear_background, atol=1.0))

    def test_emd2d_passArgsViaDict(self):

        FIXE = 10
        params = {"FIXE": FIXE}
        emd2D = EMD2D(**params)

        self.assertTrue(emd2D.FIXE == FIXE, "Received {}, Expceted {}".format(emd2D.FIXE, FIXE))

    def test_emd2d_limitImfNo(self):

        # Create image
        rows, cols = 128, 128
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
        IMFs = self.emd2d.emd(image, max_imf=max_imf)

        # It should have no more than 2 (max_imf)
        self.assertTrue(IMFs.shape[0] == max_imf)


if __name__ == "__main__":
    unittest.main()
