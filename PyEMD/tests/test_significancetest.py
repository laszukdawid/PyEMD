import numpy as np
import unittest
from PyEMD.significancetest import *

class TestCase(unittest.TestCase):
    def test_normalize(self):
        T = np.linspace(0, 1, 100)
        norm = normalize(T)
        self.assertEqual(len(T), len(norm), "Lengths must be equal")

    def test_sign_change(self):
        T = np.linspace(0, 2, 200)
        S = np.sin(2*2*np.pi*T)
        res = sign_change(normalize(S))
        self.assertEqual(type(res), int, "Default data type is int")

    def test_mean_period(self):
        T = np.linspace(0, 2, 200)
        S = np.sin(2*2*np.pi*T)
        res = mean_period(S)
        self.assertEqual(type(res), float, "Default data type is float")

    def test_energy(self):
        T = np.linspace(0, 2, 200)
        S = np.sin(2*2*np.pi*T)
        res = energy(S)
        self.assertEqual(type(res), np.float64, "Default data type is float")

    def test_significance_apriori(self):
        T = np.linspace(0, 2, 200)
        S = np.sin(2*2*np.pi*T)
        res = significance_apriori(S, 2, len(S), 0.095)
        self.assertEqual(type(res), bool, "Default data type is bool")

    def test_significance_aposteriori(self):
        T = np.linspace(0, 2, 200)
        S = np.sin(2*2*np.pi*T)
        res = significance_aposteriori(S, 2, len(S), 0.095)
        self.assertEqual(type(res), bool, "Default data type is bool")

    def test_whitenoisecheck(self):
        T = np.linspace(0, 2, 200)
        S = np.sin(2*2*np.pi*T)
        res = whitenoisecheck(S)
        self.assertEqual(type(res), dict or None, "Default data type is dict")
if __name__ == "__main__":
    unittest.main()
