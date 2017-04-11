import numpy as np
import unittest
from ..EMD import EMD

class EMDTest(unittest.TestCase):

    def test_IMF(self):
        t = np.arange(100)
        S = np.sin(3*t)

        emd = EMD()
        IMF, EXT, ITER, imfNo = emd.emd(S)

        self.assertTrue(imfNo==1)

if __name__ == "__main__":
    unittest.main()
