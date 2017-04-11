import unittest

from ..splines import *
import numpy as np

class SplineTest(unittest.TestCase):

    def test_akima(self):

        X = np.array([0,0.2,0.8])
        Y = np.array([1, 0, 1])

        x = np.array([0, 0.5, 1.])
        with self.assertRaises(Exception):
            akima(X, Y, x)

        # Test 1
        x = np.array([0, 0.2, 0.4, 0.6, 0.8])
        y = akima(X, Y, x)

        y_exp = np.array([1., 0., -0.01234568, 0.27160494, 1.])
        self.assertTrue(np.all(y_exp-y<1e-8))

        # Test 2
        X = np.array([0, 0.1, 0.5, 0.7, 0.8, 0.9])
        Y = np.array([0, 0, 1, 5, 1, 0])

        x = np.array([0, 0.2, 0.4, 0.6, 0.8])
        y = akima(X, Y, x)

        y_exp = np.array([0, 0.1275, 0.67416667, 3.13263158, 1.])
        self.assertTrue(np.all(y_exp-y<1e-8))

    def test_hermite(self):

        compare = lambda y1, y2, err: np.all(np.abs(y1-y2)<err)

        # Test 1 -- straight line
        P0, M0, P1, M1 = 0, 0, 0, 0
        t = np.arange(0, 1, 0.05)
        y = spline_hermite(t, P0, M0, P1, M1)

        self.assertTrue(compare(y,0,1e-16))

        # Test 2
        t = np.array([0, 0.21, 0.42, 0.6])
        P0, M0 = 0, 1
        P1, M1 = 1, 0

        y = spline_hermite(t, P0, M0, P1, M1)
        y_exp = np.array([0, 0.311325, 0.7966, 1.])

        self.assertTrue(compare(y_exp,y,1e-12))

        # Test 3
        t = np.array([0, 0.25, 0.5, 0.75, 1])
        P0, M0 = 1, 1
        P1, M1 = -1, -1

        y = spline_hermite(t, P0, M0, P1, M1)
        y_exp = np.array([1., 0.725, 0.05, -0.65, -1.])

        self.assertTrue(compare(y,y_exp,1e-10))

def main():
    unittest.main()

if __name__ == "__main__":
    main()
