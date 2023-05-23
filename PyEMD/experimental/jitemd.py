#!/usr/bin/python
# coding: UTF-8
#
# Author:   Dawid Laszuk
# Issues:  https://github.com/laszukdawid/PyEMD/issues
"""
Uses Numba (https://numba.pydata.org/) as a Just-in-Time (JIT)
compiler for Python, mostly Numpy. Just-in-time compilation means that
the code is compiled (machine code) during execution, and thus shows
benefit when there's plenty of repeated code or same code used a lot.

This EMD implementation is experimental as it only provides value
when there's significant amount of computation required, e.g. when
analyzing HUGE time series with a lot of internal complexity,
or reuses the instance/method many times, e.g. in a script,
iPython REPL or jupyter notebook.

Additional reason for this being experimental is that the author (me)
isn't well veristile in Numba optimization. There's definitely a lot
that can be improved. It's being added as maybe it'll be helpful or
an inspiration for others to learn something and contribute to the PyEMD.

"""

import logging
from typing import Optional, Tuple

import numba as nb
import numpy as np
from numba.types import float64, int64, unicode_type
from scipy.interpolate import Akima1DInterpolator, interp1d

FindExtremaOutput = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


default_emd_config = nb.typed.Dict.empty(
    key_type=unicode_type,
    value_type=float64,
)
default_emd_config["nbsym"] = 2.0
default_emd_config["energy_ratio_thr"] = 0.2
default_emd_config["std_thr"] = 0.2
default_emd_config["svar_thr"] = 0.001
default_emd_config["total_power_thr"] = 0.005
default_emd_config["range_thr"] = 0.001
default_emd_config["FIXE"] = 0.0
default_emd_config["FIXE_H"] = 0.0
default_emd_config["MAX_ITERATION"] = 1000.0


class JitEMD:
    def __init__(self, config=None, spline_kind="cubic", extrema_detection="simple"):
        self.config = config or default_emd_config
        self.spline_kind = spline_kind
        self.extrema_detection = extrema_detection
        self.imfs = None
        self.imfs = None
        self.residue = None

    def get_imfs_and_trend(self):
        if np.allclose(self.residue, 0):
            return self.imfs[:-1].copy(), self.imfs[-1].copy()
        return self.imfs, self.residue

    def emd(self, s, t, max_imf=-1):
        imfs = emd(
            s,
            t,
            max_imf=max_imf,
            spline_kind=self.spline_kind,
            extrema_detection=self.extrema_detection,
            config=self.config,
        )
        self.imfs = imfs[:-1, :]
        self.residue = imfs[-1:, :]
        return imfs

    def __call__(self, s, t, max_imf=-1):
        return self.emd(s, t, max_imf=max_imf)


@nb.jit(float64[:](float64[:], int64, float64[:]), nopython=True)
def np_round(x, decimals, out):
    return np.round_(x, decimals, out)


@nb.njit
def _not_duplicate(S: np.ndarray) -> np.ndarray:
    dup = np.logical_and(S[1:-1] == S[0:-2], S[1:-1] == S[2:])
    not_dup_idx = np.arange(1, len(S) - 1)[~dup]

    idx = np.empty(len(not_dup_idx) + 2, dtype=np.int64)
    idx[0] = 0
    idx[-1] = len(S) - 1
    idx[1:-1] = not_dup_idx

    return idx


@nb.jit
def cubic_spline_3pts(x, y, T):
    """
    Apparently scipy.interpolate.interp1d does not support
    cubic spline for less than 4 points.
    """
    x0, x1, x2 = x
    y0, y1, y2 = y

    x1x0, x2x1 = x1 - x0, x2 - x1
    y1y0, y2y1 = y1 - y0, y2 - y1
    _x1x0, _x2x1 = 1.0 / x1x0, 1.0 / x2x1

    m11, m12, m13 = 2 * _x1x0, _x1x0, 0
    m21, m22, m23 = _x1x0, 2.0 * (_x1x0 + _x2x1), _x2x1
    m31, m32, m33 = 0, _x2x1, 2.0 * _x2x1

    v1 = 3 * y1y0 * _x1x0 * _x1x0
    v3 = 3 * y2y1 * _x2x1 * _x2x1
    v2 = v1 + v3

    M = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
    v = np.array([v1, v2, v3]).T
    k = np.linalg.inv(M).dot(v)

    a1 = k[0] * x1x0 - y1y0
    b1 = -k[1] * x1x0 + y1y0
    a2 = k[1] * x2x1 - y2y1
    b2 = -k[2] * x2x1 + y2y1

    t = T[np.logical_and(T >= x0, T <= x2)]
    t1 = (T[np.logical_and(T >= x0, T < x1)] - x0) / x1x0
    t2 = (T[np.logical_and(T >= x1, T <= x2)] - x1) / x2x1
    t11, t22 = 1.0 - t1, 1.0 - t2

    q1 = t11 * y0 + t1 * y1 + t1 * t11 * (a1 * t11 + b1 * t1)
    q2 = t22 * y1 + t2 * y2 + t2 * t22 * (a2 * t22 + b2 * t2)
    q = np.append(q1, q2)

    return t, q


@nb.jit(nopython=False, forceobj=True)
def akima(X, Y, x):
    spl = Akima1DInterpolator(X, Y)
    return spl(x)


@nb.njit
def nb_diff(s):
    return s[1:] - s[:-1]


@nb.jit(
    nb.types.UniTuple(float64[:], 5)(float64[:], float64[:]),
    nopython=True,
)
def _find_extrema_simple(T: np.ndarray, S: np.ndarray) -> FindExtremaOutput:
    # Finds indexes of zero-crossings
    S1, S2 = S[:-1], S[1:]
    indzer = np.nonzero(S1 * S2 < 0)[0]
    if np.any(S == 0):
        indz = np.nonzero(S == 0)[0]
        if np.any(nb_diff(indz) == 1):
            zer = S == 0
            dz = nb_diff(np.append(np.append(0, zer), 0))
            debz = np.nonzero(dz == 1)[0].astype(np.float64)
            finz = (np.nonzero(dz == -1)[0] - 1).astype(np.float64)
            # indz = np.round((debz + finz) / 2)
            y = np.empty_like(debz)
            np_round((debz + finz) / 2, 0, y)
            indz = y.astype(np.int64)

        indzer = np.sort(np.append(indzer, indz))

    # Finds local extrema
    d = S2 - S1
    d1, d2 = d[:-1], d[1:]
    indmin = np.nonzero(np.logical_and(d1 * d2 < 0, d1 < 0))[0] + 1
    indmax = np.nonzero(np.logical_and(d1 * d2 < 0, d1 > 0))[0] + 1
    indmin = indmin.astype(np.int64)
    indmax = indmax.astype(np.int64)

    # When two or more points have the same value
    if np.any(d == 0):
        imax, imin = [], []

        same_values = d == 0
        dd = nb_diff(np.append(np.append(0, same_values), 0))
        _left_idx = np.nonzero(dd == 1)[0]
        _right_idx = np.nonzero(dd == -1)[0]
        if _left_idx[0] == 1:
            _left_idx = _left_idx[1:]
            _right_idx = _right_idx[1:]

        if len(_left_idx) > 0:
            if _right_idx[-1] == len(S) - 1:
                _left_idx = _left_idx[:-1]
                _right_idx = _right_idx[:-1]

        for k in range(len(_left_idx)):
            _left_value = d[_left_idx[k] - 1]
            _right_value = d[_right_idx[k]]
            _mid_value = round((_left_idx[k] + _right_idx[k]) / 2.0)
            if _left_value > 0 and _right_value < 0:
                imax.append(_mid_value)
            elif _left_value < 0 and _right_value > 0:
                imin.append(_mid_value)

        if len(imax) > 0:
            indmax = np.append(indmax, np.array(imax)).astype(np.int64)
            indmax.sort()

        if len(imin) > 0:
            indmin = np.append(indmin, np.array(imin)).astype(np.int64)
            indmin.sort()

    local_max_pos = T[indmax].astype(S.dtype)
    local_max_val = S[indmax].astype(S.dtype)
    local_min_pos = T[indmin].astype(S.dtype)
    local_min_val = S[indmin].astype(S.dtype)

    return local_max_pos, local_max_val, local_min_pos, local_min_val, indzer.astype(S.dtype)


@nb.jit(
    nb.types.Tuple((float64[:], float64[:], float64[:], float64[:], float64[:]))(float64[:], float64[:]),
    nopython=True,
)
def _find_extrema_parabol(T: np.ndarray, S: np.ndarray) -> FindExtremaOutput:
    # Finds indexes of zero-crossings
    S1, S2 = S[:-1], S[1:]
    indzer = np.nonzero(S1 * S2 < 0)[0]
    if np.any(S == 0):
        indz = np.nonzero(S == 0)[0]
        if np.any(nb_diff(indz) == 1):
            zer = S == 0
            dz = nb_diff(np.append(np.append(0, zer), 0))
            debz = (np.nonzero(dz == 1)[0]).astype(np.float64)
            finz = (np.nonzero(dz == -1)[0] - 1).astype(np.float64)
            y = np.empty_like(debz)
            np_round((debz + finz) / 2, 0, y)
            indz = y.astype(np.int64)

        indzer = np.sort(np.append(indzer, indz))

    dt = float(T[1] - T[0])
    scale = 2.0 * dt * dt

    idx = _not_duplicate(S)
    T = T[idx]
    S = S[idx]

    # p - previous
    # 0 - current
    # n - next
    Tp, T0, Tn = T[:-2], T[1:-1], T[2:]
    Sp, S0, Sn = S[:-2], S[1:-1], S[2:]
    # a = Sn + Sp - 2*S0
    # b = 2*(Tn+Tp)*S0 - ((Tn+T0)*Sp+(T0+Tp)*Sn)
    # c = Sp*T0*Tn -2*Tp*S0*Tn + Tp*T0*Sn
    TnTp, T0Tn, TpT0 = Tn - Tp, T0 - Tn, Tp - T0
    scale = Tp * Tn * Tn + Tp * Tp * T0 + T0 * T0 * Tn - Tp * Tp * Tn - Tp * T0 * T0 - T0 * Tn * Tn

    a = T0Tn * Sp + TnTp * S0 + TpT0 * Sn
    b = (S0 - Sn) * Tp**2 + (Sn - Sp) * T0**2 + (Sp - S0) * Tn**2
    c = T0 * Tn * T0Tn * Sp + Tn * Tp * TnTp * S0 + Tp * T0 * TpT0 * Sn

    a = a / scale
    b = b / scale
    c = c / scale
    a[a == 0] = 1e-14
    tVertex = -0.5 * b / a
    idx = np.logical_and(tVertex < T0 + 0.5 * (Tn - T0), tVertex >= T0 - 0.5 * (T0 - Tp))

    a, b, c = a[idx], b[idx], c[idx]
    tVertex = tVertex[idx]
    sVertex = a * tVertex * tVertex + b * tVertex + c

    local_max_pos, local_max_val = tVertex[a < 0], sVertex[a < 0]
    local_min_pos, local_min_val = tVertex[a > 0], sVertex[a > 0]

    return local_max_pos, local_max_val, local_min_pos, local_min_val, indzer.astype(local_max_val.dtype)


@nb.jit(
    nb.types.Tuple((float64[:], float64[:], float64[:], float64[:], float64[:]))(float64[:], float64[:], unicode_type),
    nopython=True,
)
def find_extrema(T: np.ndarray, S: np.ndarray, extrema_detection: str) -> FindExtremaOutput:
    """
    Returns extrema (minima and maxima) for given signal S.
    Detection and definition of the extrema depends on
    ``extrema_detection`` variable, set on initiation of EMD.

    Parameters
    ----------
    T : numpy array
        Position or time array.
    S : numpy array
        Input data S(T).

    Returns
    -------
    local_max_pos : numpy array
        Position of local maxima.
    local_max_val : numpy array
        Values of local maxima.
    local_min_pos : numpy array
        Position of local minima.
    local_min_val : numpy array
        Values of local minima.
    """
    # return _find_extrema_simple(T, S)
    if extrema_detection == "parabol":
        return _find_extrema_parabol(T, S)
    elif extrema_detection == "simple":
        return _find_extrema_simple(T, S)
    else:
        raise ValueError("Incorrect extrema detection type. Please try: 'simple' or 'parabol'.")


@nb.jit(nopython=True)
def prepare_points(
    T: np.ndarray,
    S: np.ndarray,
    max_pos: np.ndarray,
    max_val: np.ndarray,
    min_pos: np.ndarray,
    min_val: np.ndarray,
    extrema_detection,
    nbsym,
):
    """
    Performs extrapolation on edges by adding extra extrema, also known
    as mirroring signal. The number of added points depends on *nbsym*
    variable.

    Parameters
    ----------
    T : numpy array
        Position or time array.
    S : numpy array
        Input signal.
    max_pos : iterable
        Sorted time positions of maxima.
    max_val : iterable
        Signal values at max_pos positions.
    min_pos : iterable
        Sorted time positions of minima.
    min_val : iterable
        Signal values at min_pos positions.

    Returns
    -------
    max_extrema : numpy array (2 rows)
        Position (1st row) and values (2nd row) of minima.
    min_extrema : numpy array (2 rows)
        Position (1st row) and values (2nd row) of maxima.
    """
    if extrema_detection == "parabol":
        return _prepare_points_parabol(T, S, max_pos, max_val, min_pos, min_val, nbsym)
    elif extrema_detection == "simple":
        return _prepare_points_simple(T, S, max_pos, min_pos, nbsym)
    else:
        msg = "Incorrect extrema detection type. Please try: 'simple' or 'parabol'."
        raise ValueError(msg)


@nb.jit(nopython=True)
def _prepare_points_parabol(T, S, max_pos, max_val, min_pos, min_val, nbsym) -> Tuple[np.ndarray, np.ndarray]:
    # Need at least two extrema to perform mirroring
    DTYPE = S.dtype
    max_extrema = np.zeros((2, len(max_pos)), dtype=DTYPE)
    min_extrema = np.zeros((2, len(min_pos)), dtype=DTYPE)

    max_extrema[0], min_extrema[0] = max_pos, min_pos
    max_extrema[1], min_extrema[1] = max_val, min_val

    # Local variables
    end_min, end_max = len(min_pos), len(max_pos)

    ####################################
    # Left bound
    d_pos = max_pos[0] - min_pos[0]
    left_ext_max_type = d_pos < 0  # True -> max, else min

    # Left extremum is maximum
    if left_ext_max_type:
        if (S[0] > min_val[0]) and (np.abs(d_pos) > (max_pos[0] - T[0])):
            # mirror signal to first extrema
            expand_left_max_pos = 2 * max_pos[0] - max_pos[1 : nbsym + 1]
            expand_left_min_pos = 2 * max_pos[0] - min_pos[0:nbsym]
            expand_left_max_val = max_val[1 : nbsym + 1]
            expand_left_min_val = min_val[0:nbsym]
        else:
            # mirror signal to beginning
            expand_left_max_pos = 2 * T[0] - max_pos[0:nbsym]
            expand_left_min_pos = 2 * T[0] - np.append(T[0], min_pos[0 : nbsym - 1])
            expand_left_max_val = max_val[0:nbsym]
            expand_left_min_val = np.append(S[0], min_val[0 : nbsym - 1])

    # Left extremum is minimum
    else:
        if (S[0] < max_val[0]) and (np.abs(d_pos) > (min_pos[0] - T[0])):
            # mirror signal to first extrema
            expand_left_max_pos = 2 * min_pos[0] - max_pos[0:nbsym]
            expand_left_min_pos = 2 * min_pos[0] - min_pos[1 : nbsym + 1]
            expand_left_max_val = max_val[0:nbsym]
            expand_left_min_val = min_val[1 : nbsym + 1]
        else:
            # mirror signal to beginning
            expand_left_max_pos = 2 * T[0] - np.append(T[0], max_pos[0 : nbsym - 1])
            expand_left_min_pos = 2 * T[0] - min_pos[0:nbsym]
            expand_left_max_val = np.append(S[0], max_val[0 : nbsym - 1])
            expand_left_min_val = min_val[0:nbsym]

    if not expand_left_min_pos.shape:
        expand_left_min_pos, expand_left_min_val = min_pos, min_val
    if not expand_left_max_pos.shape:
        expand_left_max_pos, expand_left_max_val = max_pos, max_val

    expand_left_min = np.vstack((expand_left_min_pos[::-1], expand_left_min_val[::-1]))
    expand_left_max = np.vstack((expand_left_max_pos[::-1], expand_left_max_val[::-1]))

    ####################################
    # Right bound
    d_pos = max_pos[-1] - min_pos[-1]
    right_ext_max_type = d_pos > 0

    # Right extremum is maximum
    if not right_ext_max_type:
        if (S[-1] < max_val[-1]) and (np.abs(d_pos) > (T[-1] - min_pos[-1])):
            # mirror signal to last extrema
            idx_max = max(0, end_max - nbsym)
            idx_min = max(0, end_min - nbsym - 1)
            expand_right_max_pos = 2 * min_pos[-1] - max_pos[idx_max:]
            expand_right_min_pos = 2 * min_pos[-1] - min_pos[idx_min:-1]
            expand_right_max_val = max_val[idx_max:]
            expand_right_min_val = min_val[idx_min:-1]
        else:
            # mirror signal to end
            idx_max = max(0, end_max - nbsym + 1)
            idx_min = max(0, end_min - nbsym)
            expand_right_max_pos = 2 * T[-1] - np.append(max_pos[idx_max:], T[-1])
            expand_right_min_pos = 2 * T[-1] - min_pos[idx_min:]
            expand_right_max_val = np.append(max_val[idx_max:], S[-1])
            expand_right_min_val = min_val[idx_min:]

    # Right extremum is minimum
    else:
        if (S[-1] > min_val[-1]) and len(max_pos) > 1 and (np.abs(d_pos) > (T[-1] - max_pos[-1])):
            # mirror signal to last extremum
            idx_max = max(0, end_max - nbsym - 1)
            idx_min = max(0, end_min - nbsym)
            expand_right_max_pos = 2 * max_pos[-1] - max_pos[idx_max:-1]
            expand_right_min_pos = 2 * max_pos[-1] - min_pos[idx_min:]
            expand_right_max_val = max_val[idx_max:-1]
            expand_right_min_val = min_val[idx_min:]
        else:
            # mirror signal to end
            idx_max = max(0, end_max - nbsym)
            idx_min = max(0, end_min - nbsym + 1)
            expand_right_max_pos = 2 * T[-1] - max_pos[idx_max:]
            expand_right_min_pos = 2 * T[-1] - np.append(min_pos[idx_min:], T[-1])
            expand_right_max_val = max_val[idx_max:]
            expand_right_min_val = np.append(min_val[idx_min:], S[-1])

    if not expand_right_min_pos.shape:
        expand_right_min_pos, expand_right_min_val = min_pos, min_val
    if not expand_right_max_pos.shape:
        expand_right_max_pos, expand_right_max_val = max_pos, max_val

    expand_right_min = np.vstack((expand_right_min_pos[::-1], expand_right_min_val[::-1]))
    expand_right_max = np.vstack((expand_right_max_pos[::-1], expand_right_max_val[::-1]))

    max_extrema = np.hstack((expand_left_max, max_extrema, expand_right_max))
    min_extrema = np.hstack((expand_left_min, min_extrema, expand_right_min))

    return max_extrema, min_extrema


@nb.jit(nopython=True)
def _prepare_points_simple(
    T: np.ndarray,
    S: np.ndarray,
    max_pos: np.ndarray,
    min_pos: np.ndarray,
    nbsym: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Find indexes of pass
    ind_min = min_pos.astype(np.int64)
    ind_max = max_pos.astype(np.int64)

    # Local variables
    end_min, end_max = len(min_pos), len(max_pos)

    ####################################
    # Left bound - mirror nbsym points to the left
    if ind_max[0] < ind_min[0]:
        if S[0] > S[ind_min[0]]:
            lmax = ind_max[1 : min(end_max, nbsym + 1)][::-1]
            lmin = ind_min[0 : min(end_min, nbsym + 0)][::-1]
            lsym = ind_max[0]
        else:
            lmax = ind_max[0 : min(end_max, nbsym)][::-1]
            lmin = np.append(ind_min[0 : min(end_min, nbsym - 1)][::-1], 0)
            lsym = 0
    else:
        if S[0] < S[ind_max[0]]:
            lmax = ind_max[0 : min(end_max, nbsym + 0)][::-1]
            lmin = ind_min[1 : min(end_min, nbsym + 1)][::-1]
            lsym = ind_min[0]
        else:
            lmax = np.append(ind_max[0 : min(end_max, nbsym - 1)][::-1], 0)
            lmin = ind_min[0 : min(end_min, nbsym)][::-1]
            lsym = 0

    ####################################
    # Right bound - mirror nbsym points to the right
    if ind_max[-1] < ind_min[-1]:
        if S[-1] < S[ind_max[-1]]:
            rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
            rmin = ind_min[max(end_min - nbsym - 1, 0) : -1][::-1]
            rsym = ind_min[-1]
        else:
            rmax = np.append(ind_max[max(end_max - nbsym + 1, 0) :], len(S) - 1)[::-1]
            rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
            rsym = len(S) - 1
    else:
        if S[-1] > S[ind_min[-1]]:
            rmax = ind_max[max(end_max - nbsym - 1, 0) : -1][::-1]
            rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
            rsym = ind_max[-1]
        else:
            rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
            rmin = np.append(ind_min[max(end_min - nbsym + 1, 0) :], len(S) - 1)[::-1]
            rsym = len(S) - 1

    # In case any array missing
    if not lmin.size:
        lmin = ind_min
    if not rmin.size:
        rmin = ind_min
    if not lmax.size:
        lmax = ind_max
    if not rmax.size:
        rmax = ind_max

    # Mirror points
    tlmin = 2 * T[lsym] - T[lmin]
    tlmax = 2 * T[lsym] - T[lmax]
    trmin = 2 * T[rsym] - T[rmin]
    trmax = 2 * T[rsym] - T[rmax]

    # If mirrored points are not outside passed time range.
    if tlmin[0] > T[0] or tlmax[0] > T[0]:
        if lsym == ind_max[0]:
            lmax = ind_max[0 : min(end_max, nbsym)][::-1]
        else:
            lmin = ind_min[0 : min(end_min, nbsym)][::-1]

        if lsym == 0:
            raise Exception("Left edge BUG")

        lsym = 0
        tlmin = 2 * T[lsym] - T[lmin]
        tlmax = 2 * T[lsym] - T[lmax]

    if trmin[-1] < T[-1] or trmax[-1] < T[-1]:
        if rsym == ind_max[-1]:
            rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
        else:
            rmin = ind_min[max(end_min - nbsym, 0) :][::-1]

        if rsym == len(S) - 1:
            raise Exception("Right edge BUG")

        rsym = len(S) - 1
        trmin = 2 * T[rsym] - T[rmin]
        trmax = 2 * T[rsym] - T[rmax]

    zlmax = S[lmax]
    zlmin = S[lmin]
    zrmax = S[rmax]
    zrmin = S[rmin]

    tmin = np.append(tlmin, np.append(T[ind_min], trmin))
    tmax = np.append(tlmax, np.append(T[ind_max], trmax))
    zmin = np.append(zlmin, np.append(S[ind_min], zrmin))
    zmax = np.append(zlmax, np.append(S[ind_max], zrmax))

    max_extrema = np.vstack((tmax, zmax))
    min_extrema = np.vstack((tmin, zmin))

    # For posterity:
    #  I tried with np.delete and np.vstack([ ]) but both didn't work.
    #  np.delete works only with 2 args, and vstack had problem with list comphr.
    max_dup_idx = np.where(max_extrema[0, 1:] == max_extrema[0, :-1])[0]
    if len(max_dup_idx):
        for col_idx in max_dup_idx:
            max_extrema = np.hstack((max_extrema[:, :col_idx], max_extrema[:, col_idx + 1 :]))

    min_dup_idx = np.where(min_extrema[0, 1:] == min_extrema[0, :-1])[0]
    if len(min_dup_idx):
        for col_idx in min_dup_idx:
            min_extrema = np.hstack((min_extrema[:, :col_idx], min_extrema[:, col_idx + 1 :]))

    return max_extrema, min_extrema


@nb.jit(nopython=False, forceobj=True)
def spline_points(T: np.ndarray, extrema: np.ndarray, spline_kind: str) -> Tuple[np.ndarray, np.ndarray]:
    dtype = extrema.dtype
    kind = spline_kind.lower()
    t = T[np.logical_and(T >= extrema[0, 0], T <= extrema[0, -1])]

    if kind == "akima":
        return t, akima(extrema[0], extrema[1], t)

    elif kind == "cubic":
        if extrema.shape[1] > 3:
            interpolation = interp1d(extrema[0], extrema[1], kind=kind)
            interpolated: np.ndarray = interpolation(t)
            return t, interpolated.astype(dtype)
        else:
            return cubic_spline_3pts(extrema[0], extrema[1], t)

    elif kind in ["slinear", "quadratic", "linear"]:
        interpolation = interp1d(extrema[0], extrema[1], kind=kind)
        interpolated: np.ndarray = interpolation(t)
        return T, interpolated.astype(dtype)

    else:
        raise ValueError("No such interpolation method!")


@nb.jit(
    nb.types.UniTuple(float64[:, :], 2)(float64[:], float64[:], int64, unicode_type),
    nopython=True,
)
def extract_max_min_extrema(
    T: np.ndarray,
    S: np.ndarray,
    nbsym: int,
    extrema_detection: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts top and bottom envelopes based on the signal,
    which are constructed based on maxima and minima, respectively.

    Parameters
    ----------
    T : numpy array
        Position or time array.
    S : numpy array
        Input data S(T).

    Returns
    -------
    max_spline : numpy array
        Spline spanned on S maxima.
    min_spline : numpy array
        Spline spanned on S minima.
    max_extrema : numpy array
        Points indicating local maxima.
    min_extrema : numpy array
        Points indicating local minima.
    """

    # Get indexes of extrema
    ext_res = find_extrema(T, S, extrema_detection)
    max_pos, max_val = ext_res[0], ext_res[1]
    min_pos, min_val = ext_res[2], ext_res[3]

    if len(max_pos) + len(min_pos) < 3:
        minus_one = -1 * np.ones((1, 1), dtype=S.dtype)
        return minus_one.copy(), minus_one.copy()

    #########################################
    # Extrapolation of signal (over boundaries)
    max_extrema, min_extrema = prepare_points(T, S, max_pos, max_val, min_pos, min_val, extrema_detection, nbsym)
    return max_extrema, min_extrema


@nb.njit
def end_condition(S: np.ndarray, IMF: np.ndarray, range_thr: float, total_power_thr: float) -> bool:
    """Tests for end condition of whole EMD. The procedure will stop if:

    * Absolute amplitude (max - min) is below *range_thr* threshold, or
    * Metric L1 (mean absolute difference) is below *total_power_thr* threshold.

    Parameters
    ----------
    S : numpy array
        Original signal on which EMD was performed.
    IMF : numpy 2D array
        Set of IMFs where each row is IMF. Their order is not important.

    Returns
    -------
    end : bool
        Whether sifting is finished.
    """
    # When to stop EMD
    tmp = S - np.sum(IMF, axis=0)

    if np.max(tmp) - np.min(tmp) < range_thr:
        return True

    if np.sum(np.abs(tmp)) < total_power_thr:
        return True

    return False


@nb.jit
def check_imf(
    imf_new: np.ndarray,
    imf_old: np.ndarray,
    eMax: np.ndarray,
    eMin: np.ndarray,
    svar_thr: float,
    std_thr: float,
    energy_ratio_thr: float,
) -> bool:
    """
    Huang criteria for **IMF** (similar to Cauchy convergence test).
    Signal is an IMF if consecutive siftings do not affect signal
    in a significant manner.
    """
    # local max are >0 and local min are <0
    # if np.any(eMax[1] < 0) or np.any(eMin[1] > 0):
    if np.sum(eMax[1] < 0) + np.sum(eMin[1] > 0) > 0:
        return False

    # Convergence
    if np.sum(imf_new**2) < 1e-10:
        return False

    # Precompute values
    imf_diff = imf_new - imf_old
    imf_diff_sqrd_sum = np.sum(imf_diff * imf_diff)

    # Scaled variance test
    svar = imf_diff_sqrd_sum / (max(imf_old) - min(imf_old))
    if svar < svar_thr:
        return True

    # Standard deviation test
    std = np.sum((imf_diff / imf_new) ** 2)
    if std < std_thr:
        return True

    energy_ratio = imf_diff_sqrd_sum / np.sum(imf_old * imf_old)
    if energy_ratio < energy_ratio_thr:
        return True

    return False


# @nb.jit
def _common_dtype(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Casts inputs (x, y) into a common numpy DTYPE."""
    dtype = np.find_common_type([x.dtype, y.dtype], [])
    if x.dtype != dtype:
        x = x.astype(dtype)
    if y.dtype != dtype:
        y = y.astype(dtype)
    return x, y


@nb.jit
def _normalize_time(t: np.ndarray) -> np.ndarray:
    """
    Normalize time array so that it doesn't explode on tiny values.
    Returned array starts with 0 and the smallest increase is by 1.
    """
    d = nb_diff(t)
    assert np.all(d != 0), "All time domain values needs to be unique"
    return (t - t[0]) / np.min(d)


@nb.jit(
    float64[:, :](
        float64[:],
        float64[:],
        nb.optional(int64),
        nb.optional(unicode_type),
        nb.optional(unicode_type),
        nb.optional(nb.typeof(default_emd_config)),
    ),
    nopython=False,
    forceobj=True,
)
def emd(
    S: np.ndarray,
    T: np.ndarray,
    max_imf: int = -1,
    spline_kind: str = "cubic",
    extrema_detection: str = "simple",
    config=default_emd_config,
) -> np.ndarray:
    # if T is not None and len(S) != len(T):
    #     raise ValueError("Time series have different sizes: len(S) -> {} != {} <- len(T)".format(len(S), len(T)))
    #     return None

    # if T is None or config["extrema_detection"] == "simple":
    #     T = get_timeline(len(S), S.dtype)

    # Normalize T so that it doesn't explode
    # T = _normalize_time(T)

    # Make sure same types are dealt
    # S, T = _common_dtype(S, T)
    MAX_ITERATION = config["MAX_ITERATION"]
    FIXE = config["FIXE"]
    FIXE_H = config["FIXE_H"]
    nbsym = config["nbsym"]
    svar_thr, std_thr, energy_ratio_thr = config["svar_thr"], config["std_thr"], config["energy_ratio_thr"]
    range_thr, total_power_thr = config["range_thr"], config["total_power_thr"]

    DTYPE = S.dtype
    N = len(S)

    residue = S.astype(DTYPE)
    imf = np.zeros(len(S), dtype=DTYPE)
    imf_old = np.nan

    if S.shape != T.shape:
        raise ValueError("Position or time array should be the same size as signal.")

    # Create arrays
    imfNo = 0
    extNo = -1
    IMF = np.empty((imfNo, N))  # Numpy container for IMF
    finished = False

    while not finished:
        residue[:] = S - np.sum(IMF[:imfNo], axis=0)
        imf = residue.copy()
        mean = np.zeros(len(S), dtype=DTYPE)

        # Counters
        n = 0  # All iterations for current imf.
        n_h = 0  # counts when |#zero - #ext| <=1

        while True:
            n += 1
            if n >= MAX_ITERATION:
                break

            ext_res = find_extrema(T, imf, extrema_detection)
            max_pos, min_pos, indzer = ext_res[0], ext_res[2], ext_res[4]
            extNo = len(min_pos) + len(max_pos)
            nzm = len(indzer)

            if extNo > 2:
                max_extrema, min_extrema = extract_max_min_extrema(T, imf, nbsym, extrema_detection)
                _, max_env = spline_points(T, max_extrema, spline_kind)
                _, min_env = spline_points(T, min_extrema, spline_kind)
                mean = 0.5 * (max_env + min_env)

                imf_old = imf.copy()
                imf = imf - mean

                # Fix number of iterations
                if FIXE:
                    if n >= FIXE:
                        break

                # Fix number of iterations after number of zero-crossings
                # and extrema differ at most by one.
                elif FIXE_H:
                    max_pos, _, min_pos, _, ind_zer = find_extrema(T, imf, extrema_detection)
                    extNo = len(max_pos) + len(min_pos)
                    nzm = len(ind_zer)

                    if n == 1:
                        continue

                    # If proto-IMF add one, or reset counter otherwise
                    n_h = n_h + 1 if abs(extNo - nzm) < 2 else 0

                    # STOP
                    if n_h >= FIXE_H:
                        break

                # Stops after default stopping criteria are met
                else:
                    ext_res = find_extrema(T, imf, extrema_detection)
                    max_pos, _, min_pos, _, ind_zer = ext_res
                    extNo = len(max_pos) + len(min_pos)
                    nzm = len(ind_zer)

                    if imf_old is np.nan:
                        continue

                    f1 = check_imf(imf, imf_old, max_extrema, min_extrema, svar_thr, std_thr, energy_ratio_thr)
                    f2 = abs(extNo - nzm) < 2

                    # STOP
                    if f1 and f2:
                        break

            else:  # Less than 2 ext, i.e. trend
                finished = True
                break
        # END OF IMF SIFTING

        IMF = np.vstack((IMF, imf.copy()))
        imfNo += 1

        if end_condition(S, IMF, range_thr, total_power_thr) or imfNo == max_imf:
            finished = True
            break

    # If the last sifting had 2 or less extrema then that's a trend (residue)
    if extNo <= 2:
        IMF = IMF[:-1]

    # Saving imfs and residue for external references
    imfs = IMF.copy()
    residue = S - np.sum(imfs, axis=0)

    # If residue isn't 0 then add it to the output
    if not np.allclose(residue, 0):
        IMF = np.vstack((IMF, residue))

    return IMF


def get_timeline(range_max: int, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Returns timeline array for requirements.

    Parameters
    ----------
    range_max : int
        Largest value in range. Assume `range(range_max)`. Commonly that's length of the signal.
    dtype : np.dtype
        Minimal definition type. Returned timeline will have dtype that's the same or with higher byte size.

    """
    timeline = np.arange(0, range_max, dtype=dtype)
    if timeline[-1] != range_max - 1:
        inclusive_dtype = smallest_inclusive_dtype(timeline.dtype, range_max)
        timeline = np.arange(0, range_max, dtype=inclusive_dtype)
    return timeline


def smallest_inclusive_dtype(ref_dtype: np.dtype, ref_value) -> np.dtype:
    """Returns a numpy dtype with the same base as reference dtype (ref_dtype)
    but with the range that includes reference value (ref_value).

    Parameters
    ----------
    ref_dtype : dtype
         Reference dtype. Used to select the base, i.e. int or float, for returned type.
    ref_value : value
        A value which needs to be included in returned dtype. Value will be typically int or float.

    """
    # Integer path
    if np.issubdtype(ref_dtype, np.integer):
        for dtype in [np.uint16, np.uint32, np.uint64]:
            if ref_value < np.iinfo(dtype).max:
                return dtype
        max_val = np.iinfo(np.uint32).max
        raise ValueError("Requested too large integer range. Exceeds max( uint64 ) == '{}.".format(max_val))

    # Integer path
    if np.issubdtype(ref_dtype, np.floating):
        for dtype in [np.float16, np.float32, np.float64]:
            if ref_value < np.finfo(dtype).max:
                return dtype
        max_val = np.finfo(np.float64).max
        raise ValueError("Requested too large integer range. Exceeds max( float64 ) == '{}.".format(max_val))

    raise ValueError("Unsupported dtype '{}'. Only intX and floatX are supported.".format(ref_dtype))


###################################################


if __name__ == "__main__":
    import pylab as plt

    # Logging options
    logging.basicConfig(level=logging.DEBUG)

    # EMD options
    max_imf = -1
    DTYPE = np.float64

    # Signal options
    N = 400
    tMin, tMax = 0, 2 * np.pi
    T = np.linspace(tMin, tMax, N, dtype=DTYPE)

    S = np.sin(20 * T * (1 + 0.2 * T)) + T**2 + np.sin(13 * T)
    S = S.astype(DTYPE)
    print("Input S.dtype: " + str(S.dtype))

    # Prepare and run EMD
    config = EmdConfig()
    imfs = emd(config, S, T, max_imf)
    imfNo = imfs.shape[0]

    # Plot results
    c = 1
    r = np.ceil((imfNo + 1) / c)

    plt.ioff()
    plt.subplot(r, c, 1)
    plt.plot(T, S, "r")
    plt.xlim((tMin, tMax))
    plt.title("Original signal")

    for num in range(imfNo):
        plt.subplot(r, c, num + 2)
        plt.plot(T, imfs[num], "g")
        plt.xlim((tMin, tMax))
        plt.ylabel("Imf " + str(num + 1))

    plt.tight_layout()
    plt.show()
