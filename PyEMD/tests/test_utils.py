import unittest

import numpy as np

from PyEMD.utils import deduce_common_type, get_timeline, unify_types


class MyTestCase(unittest.TestCase):
    def test_get_timeline_default_dtype(self):
        S = np.random.random(100)
        T = get_timeline(len(S))

        self.assertEqual(len(T), len(S), "Lengths must be equal")
        self.assertEqual(T.dtype, np.int64, "Default dtype is np.int64")
        self.assertEqual(T[-1], len(S) - 1, "Range is kept")

    def test_get_timeline_signal_dtype(self):
        S = np.random.random(100)
        T = get_timeline(len(S), dtype=S.dtype)

        self.assertEqual(len(T), len(S), "Lengths must be equal")
        self.assertEqual(T.dtype, S.dtype, "Dtypes must be equal")
        self.assertEqual(T[-1], len(S) - 1, "Range is kept")

    def test_get_timeline_does_not_overflow_int16(self):
        S = np.random.randint(100, size=(np.iinfo(np.int16).max + 10,), dtype=np.int16)
        T = get_timeline(len(S), dtype=S.dtype)

        self.assertGreater(len(S), np.iinfo(S.dtype).max, "Length of the signal is greater than its type max value")
        self.assertEqual(len(T), len(S), "Lengths must be equal")
        self.assertEqual(T[-1], len(S) - 1, "Range is kept")
        self.assertEqual(T.dtype, np.uint16, "UInt16 is the min type that matches requirements")

    def test_deduce_common_types(self):
        self.assertEqual(deduce_common_type(np.int16, np.int32), np.int32)
        self.assertEqual(deduce_common_type(np.int32, np.int16), np.int32)
        self.assertEqual(deduce_common_type(np.int32, np.int32), np.int32)
        self.assertEqual(deduce_common_type(np.float32, np.float64), np.float64)
    
    def test_unify_types(self):
        x = np.array([1, 2, 3], dtype=np.int16)
        y = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        x, y = unify_types(x, y)
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
