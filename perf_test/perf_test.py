from PyEMD import EMD

import numpy as np

def test_prepare_points_simple(benchmark):
    emd = EMD()
    T_max = 50
    T = np.linspace(0, T_max, 10000)
    S = np.sin(T*2*np.pi)
    # max_pos = np.array([0.25, 1.25, 2.25, 3.25], dtype=np.float32)
    max_pos = np.arange(T_max - 1, dtype=np.float32) + 0.25
    max_val = np.ones(T_max-1, dtype=np.float32)
    min_pos = np.arange(1, T_max, dtype=np.float32) - 0.25
    min_val = (-1)*np.ones(T_max-1, dtype=np.float32)
    # min_pos = np.array([0.75, 1.75, 2.75, 3.75], dtype=np.float32)
    # min_val = np.array([-1, -1, -1, -1], dtype=np.float32)

    benchmark.pedantic(emd.prepare_points_parabol, args=(T, S, max_pos, max_val, min_pos, min_val), iterations=100, rounds=100)
