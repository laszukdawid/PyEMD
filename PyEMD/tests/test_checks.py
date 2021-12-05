"""Tests for checks.py."""
import unittest

import numpy as np

from PyEMD.checks import energy, mean_period, significance_aposteriori, significance_apriori, whitenoise_check


class TestCase(unittest.TestCase):
    """Test cases."""

    def test_mean_period(self):
        """Test to check if mean period output is correct."""
        T = np.linspace(0, 2, 100)
        S = np.sin(2 * np.pi * T)
        res = mean_period(S)
        self.assertEqual(type(res), float, "Default data type is float")
        self.assertTrue(res > 0, "mean-period cannot be zero")

    def test_mean_period_zero_peaks(self):
        """Tect to check if mean period function can handle zero peaks."""
        T = np.linspace(0, 2, 100)
        res = mean_period(T)
        self.assertEqual(res, len(T), "mean-period is same as signal length in case of monotonic curve")

    def test_energy(self):
        """Test to check if energy of signal is being computed properly."""
        T = np.linspace(0, 2, 200)
        S = np.sin(2 * 2 * np.pi * T)
        res = energy(S)
        self.assertEqual(type(res), np.float64, "Default data type is float")

    def test_significance_apriori(self):
        """a priori significance test."""
        T = np.linspace(0, 2, 200)
        S = np.sin(2 * 2 * np.pi * T)
        energy_density = energy(S) / len(S)
        res = significance_apriori(energy_density, 2, len(S), 0.9)
        self.assertEqual(type(res), bool, "Default data type is bool")

    def test_significance_aposteriori(self):
        """a posteriori significance test."""
        T = np.linspace(0, 2, 200)
        S = np.sin(2 * 2 * np.pi * T)
        energy_density = energy(S) / len(S)
        res = significance_aposteriori(energy_density, 2, len(S), 0.9)
        self.assertEqual(type(res), bool, "Default data type is bool")

    def test_whitenoise_check_apriori(self):
        """a priori whitenoise_check."""
        T = [np.linspace(0, i, 200) for i in range(5, 0, -1)]
        S = np.array([list(np.sin(2 * 2 * np.pi * i)) for i in T])
        res = whitenoise_check(S, test_name="apriori")
        self.assertEqual(type(res), dict or None, "Default data type is dict")

    def test_whitenoise_check_apriori_alpha(self):
        """a priori whitenoise_check with custom alpha."""
        T = [np.linspace(0, i, 200) for i in range(5, 0, -1)]
        S = np.array([list(np.sin(2 * 2 * np.pi * i)) for i in T])
        res = whitenoise_check(S, test_name="apriori", alpha=0.99)
        self.assertEqual(type(res), dict or None, "Default data type is dict")

    def test_whitenoise_check_alpha(self):
        """a posteriori whitenoise check with custom alpha value."""
        T = [np.linspace(0, i, 200) for i in range(5, 0, -1)]
        S = np.array([list(np.sin(2 * 2 * np.pi * i)) for i in T])
        res = whitenoise_check(S, alpha=0.9)
        self.assertEqual(type(res), dict or None, "Default data type is dict")

    def test_whitenoise_check_rescaling_imf(self):
        """a posteriori whitenoise check with custom rescaling imf."""
        T = [np.linspace(0, i, 200) for i in range(5, 0, -1)]
        S = np.array([list(np.sin(2 * 2 * np.pi * i)) for i in T])
        res = whitenoise_check(S, rescaling_imf=2)
        self.assertEqual(type(res), dict or None, "Default data type is dict")

    def test_whitenoise_check_nan_values(self):
        """whitenoise check with nan in IMF."""
        S = np.array([np.full(100, np.NaN) for i in range(5, 0, -1)])
        res = whitenoise_check(S)
        self.assertEqual(res, None, "Input NaN returns None")

    def test_invalid_alpha(self):
        """Test if invalid alpha return AssertionError."""
        S = np.array([np.full(100, np.NaN) for i in range(5, 0, -1)])
        self.assertRaises(AssertionError, whitenoise_check, S, alpha=1)
        self.assertRaises(AssertionError, whitenoise_check, S, alpha=0)
        self.assertRaises(AssertionError, whitenoise_check, S, alpha=-10)
        self.assertRaises(AssertionError, whitenoise_check, S, alpha=2)
        self.assertRaises(AssertionError, whitenoise_check, S, alpha="0.5")

    def test_invalid_test_name(self):
        """Test if invalid test return AssertionError."""
        S = np.random.random((5, 100))
        self.assertRaises(AssertionError, whitenoise_check, S, test_name="apri")
        self.assertRaises(AssertionError, whitenoise_check, S, test_name="apost")
        self.assertRaises(AssertionError, whitenoise_check, S, test_name=None)

    def test_invalid_input_type(self):
        """Test if invalid input type return AssertionError."""
        S = [np.full(100, np.NaN) for i in range(5, 0, -1)]
        self.assertRaises(AssertionError, whitenoise_check, S)
        self.assertRaises(AssertionError, whitenoise_check, 1)
        self.assertRaises(AssertionError, whitenoise_check, 1.2)
        self.assertRaises(AssertionError, whitenoise_check, "[1,2,3,4,5]")

    def test_invalid_rescaling_imf(self):
        """Test if invalid rescaling imf return AssertionError."""
        T = [np.linspace(0, i, 200) for i in range(5, 0, -1)]
        S = np.array([list(np.sin(2 * 2 * np.pi * i)) for i in T])
        self.assertRaises(AssertionError, whitenoise_check, S, rescaling_imf=10)
        self.assertRaises(AssertionError, whitenoise_check, S, rescaling_imf=1.2)

    def test_empty_input_imf(self):
        """Test if empty IMF input return AssertionError."""
        T1 = np.array([[], []])
        T2 = np.array([])
        res1 = whitenoise_check(T1)
        res2 = whitenoise_check(T2)
        self.assertEqual(res1, None, "Empty input returns None")
        self.assertEqual(res2, None, "Empty input returns None")


if __name__ == "__main__":
    unittest.main()
