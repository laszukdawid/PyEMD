#!/usr/bin/python
# Coding: UTF-8

from __future__ import print_function

import logging
import numpy as np
import os
import time
import unittest

from PyEMD import EMD

class PerformanceTest(unittest.TestCase):

    logger = logging.getLogger(__name__)

    @staticmethod
    def _timeit(fn, args, N=10):
        avg_t = 0
        for _ in range(N):
            t0 = time.time()
            fn(*args)
            t1 = time.time()
            avg_t += t1-t0
        return avg_t/N

    @unittest.skip("Performance test should be run on server. Skipping until getting one.")
    def test_EMD_max_execution_time(self):
        t_min, t_max = 0, 1
        N = 100
        T = np.linspace(t_min, t_max, N)
        all_test_signals = []

        # These are local values. I'd be veeerrry surprised if everyone had such performance.
        # In case your test is failing: ignore and carry on.
        expected_times = np.array([0.015, 0.04, 0.04, 0.05, 0.04, 0.05, 0.05, 0.05, 0.03, 0.05])
        received_times = [0]*len(expected_times)

        # Detect whether run on Travis CI
        # Performance test should be run on same machine with
        # same setting. Travis cannot guarantee that.
        if "TRAVIS" in os.environ:
            expected_times *= 10 # Conservative.

        all_w = np.arange(10,20)
        for w in all_w:
            signal = np.sin(w*2*np.pi*T)
            signal[:] = signal[:] + 2*np.cos(5*2*np.pi*T)
            all_test_signals.append(signal)

        emd = EMD()
        emd.FIXE = 10

        for idx, signal in enumerate(all_test_signals):
            avg_t = self._timeit(emd.emd, (signal, T), N=15)

            self.logger.info("{}. t = {:.4} (exp. {})".format(idx, avg_t, expected_times[idx]))
            received_times[idx] = avg_t

       # allclose = np.allclose(received_times, expected_times, atol=1e-2)
       # self.assertTrue(allclose)

        less_than = received_times <= expected_times
        self.assertTrue(np.all(less_than))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
