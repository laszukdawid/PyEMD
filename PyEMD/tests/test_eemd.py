#!/usr/bin/python
# Coding: UTF-8

from __future__ import print_function

import numpy as np
from PyEMD import EEMD
import unittest

class ExtremaTest(unittest.TestCase):

    def test_general_eemd(self):
        S = np.random.random(100)

        eemd = EEMD()
        eemd.trials = 10
        eemd.eemd(S)

if __name__ == "__main__":
    unittest.main()
