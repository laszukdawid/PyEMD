#!/usr/bin/python
# Coding: UTF-8

from __future__ import print_function

import logging
import numpy as np
import time
import unittest

from PyEMD import EMD

class PerformanceTest(unittest.TestCase):

    logger = logging.getLogger(__name__)

    def _timeit(self, fn, signal, T, N=10):
        avg_t = 0
        for n in range(N):
            t0 = time.time()
            IMF = fn(signal, T)
            t1 = time.time()
            avg_t += t1-t0
        return avg_t/N

    def test_EMD_max_execution_time(self):
        t_min, t_max = 0, 1
        N = 100
        T = np.linspace(t_min, t_max, N)
        all_test_signals = []

        # These are local values. I'd be veeerrry surprised if everyone had such performance.
        # In case your test is failing: ignore and carry on.
        expected_times = [0.012, 0.039, 0.038, 0.050, 0.038, 0.049, 0.048, 0.050, 0.027, 0.050]
        received_times = [0]*len(expected_times)

        all_w = np.arange(10,20)
        for w in all_w:
            signal = np.sin(w*2*np.pi*T)
            signal[:] = signal[:] + 2*np.cos(5*2*np.pi*T)
            all_test_signals.append(signal)

        emd = EMD()
        emd.FIXE = 10

        for idx, signal in enumerate(all_test_signals):
            avg_t = self._timeit(emd.emd, signal, T, N=10)

            self.logger.info("{}. t = {:.4} (exp. {})".format(idx, avg_t, expected_times[idx]))
            received_times[idx] = avg_t

       # allclose = np.allclose(received_times, expected_times, atol=1e-2)
       # self.assertTrue(allclose)

        less_than = received_times <= expected_times
        self.assertTrue(np.all(less_than))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
