#!/usr/bin/python
# coding: UTF-8
#
# Author:   Dawid Laszuk
# Contact:  https://github.com/laszukdawid/PyEMD/issues
#
# Feel free to contact for any information.

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from PyEMD.splines import akima, cubic_spline_3pts
from PyEMD.utils import get_timeline

FindExtremaOutput = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class EMD:
    """
    .. _EMD:

    **Empirical Mode Decomposition**

    Method of decomposing signal into Intrinsic Mode Functions (IMFs)
    based on algorithm presented in Huang et al. [Huang1998]_.

    Algorithm was validated with Rilling et al. [Rilling2003]_ Matlab's version from 3.2007.

    Threshold which control the goodness of the decomposition:
        * `std_thr` --- Test for the proto-IMF how variance changes between siftings.
        * `svar_thr` -- Test for the proto-IMF how energy changes between siftings.
        * `total_power_thr` --- Test for the whole decomp how much of energy is solved.
        * `range_thr` --- Test for the whole decomp whether the difference is tiny.


    References
    ----------
    .. [Huang1998] N. E. Huang et al., "The empirical mode decomposition and the
        Hilbert spectrum for non-linear and non stationary time series
        analysis", Proc. Royal Soc. London A, Vol. 454, pp. 903-995, 1998
    .. [Rilling2003] G. Rilling, P. Flandrin and P. Goncalves, "On Empirical Mode
        Decomposition and its algorithms", IEEE-EURASIP Workshop on
        Nonlinear Signal and Image Processing NSIP-03, Grado (I), June 2003

    Examples
    --------
    >>> import numpy as np
    >>> T = np.linspace(0, 1, 100)
    >>> S = np.sin(2*2*np.pi*T)
    >>> emd = EMD(extrema_detection='parabol')
    >>> IMFs = emd.emd(S)
    >>> IMFs.shape
    (1, 100)
    """

    logger = logging.getLogger(__name__)

    def __init__(self, spline_kind: str = "cubic", nbsym: int = 2, **kwargs):
        """Initiate *EMD* instance.

        Configuration, such as threshold values, can be passed as kwargs (keyword arguments).

        Parameters
        ----------
        FIXE : int (default: 0)
        FIXE_H : int (default: 0)
        MAX_ITERATION : int (default 1000)
            Maximum number of iterations per single sifting in EMD.
        energy_ratio_thr : float (default: 0.2)
            Threshold value on energy ratio per IMF check.
        std_thr float : (default 0.2)
            Threshold value on standard deviation per IMF check.
        svar_thr float : (default 0.001)
            Threshold value on scaled variance per IMF check.
        total_power_thr : float (default 0.005)
            Threshold value on total power per EMD decomposition.
        range_thr : float (default 0.001)
            Threshold for amplitude range (after scaling) per EMD decomposition.
        extrema_detection : str (default 'simple')
            Method used to finding extrema.
        DTYPE : np.dtype (default np.float64)
            Data type used.

        Examples
        --------
        >>> emd = EMD(std_thr=0.01, range_thr=0.05)

        """
        # Declare constants
        self.energy_ratio_thr = float(kwargs.get("energy_ratio_thr", 0.2))
        self.std_thr = float(kwargs.get("std_thr", 0.2))
        self.svar_thr = float(kwargs.get("svar_thr", 0.001))
        self.total_power_thr = float(kwargs.get("total_power_thr", 0.005))
        self.range_thr = float(kwargs.get("range_thr", 0.001))

        self.nbsym = int(kwargs.get("nbsym", nbsym))
        self.scale_factor = float(kwargs.get("scale_factor", 1.0))

        self.spline_kind = spline_kind
        self.extrema_detection = kwargs.get("extrema_detection", "simple")  # simple, parabol
        assert self.extrema_detection in (
            "simple",
            "parabol",
        ), "Only 'simple' and 'parabol' values supported"

        self.DTYPE = kwargs.get("DTYPE", np.float64)
        self.FIXE = int(kwargs.get("FIXE", 0))
        self.FIXE_H = int(kwargs.get("FIXE_H", 0))

        self.MAX_ITERATION = int(kwargs.get("MAX_ITERATION", 1000))

        # Instance global declaration
        self.imfs = None  # Optional[np.ndarray]
        self.residue = None  # Optional[np.ndarray]

    def __call__(self, S: np.ndarray, T: Optional[np.ndarray] = None, max_imf: int = -1) -> np.ndarray:
        return self.emd(S, T=T, max_imf=max_imf)

    def extract_max_min_spline(
        self, T: np.ndarray, S: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        ext_res = self.find_extrema(T, S)
        max_pos, max_val = ext_res[0], ext_res[1]
        min_pos, min_val = ext_res[2], ext_res[3]

        if len(max_pos) + len(min_pos) < 3:
            return [-1] * 4  # TODO: Fix this. Doesn't match the signature.

        #########################################
        # Extrapolation of signal (over boundaries)
        max_extrema, min_extrema = self.prepare_points(T, S, max_pos, max_val, min_pos, min_val)

        _, max_spline = self.spline_points(T, max_extrema)
        _, min_spline = self.spline_points(T, min_extrema)

        return max_spline, min_spline, max_extrema, min_extrema

    def prepare_points(
        self,
        T: np.ndarray,
        S: np.ndarray,
        max_pos: np.ndarray,
        max_val: np.ndarray,
        min_pos: np.ndarray,
        min_val: np.ndarray,
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
        if self.extrema_detection == "parabol":
            return self._prepare_points_parabol(T, S, max_pos, max_val, min_pos, min_val)
        elif self.extrema_detection == "simple":
            return self._prepare_points_simple(T, S, max_pos, max_val, min_pos, min_val)
        else:
            msg = "Incorrect extrema detection type. Please try: 'simple' or 'parabol'."
            raise ValueError(msg)

    def _prepare_points_parabol(self, T, S, max_pos, max_val, min_pos, min_val) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs mirroring on signal which extrema do not necessarily
        belong on the position array.

        See :meth:`EMD.prepare_points`.
        """

        # Need at least two extrema to perform mirroring
        max_extrema = np.zeros((2, len(max_pos)), dtype=self.DTYPE)
        min_extrema = np.zeros((2, len(min_pos)), dtype=self.DTYPE)

        max_extrema[0], min_extrema[0] = max_pos, min_pos
        max_extrema[1], min_extrema[1] = max_val, min_val

        # Local variables
        nbsym = self.nbsym
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

    def _prepare_points_simple(
        self,
        T: np.ndarray,
        S: np.ndarray,
        max_pos: np.ndarray,
        max_val: Optional[np.ndarray],
        min_pos: np.ndarray,
        min_val: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs mirroring on signal which extrema can be indexed on
        the position array.

        See :meth:`EMD.prepare_points`.
        """

        # Find indexes of pass
        ind_min = min_pos.astype(int)
        ind_max = max_pos.astype(int)

        # Local variables
        nbsym = self.nbsym
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

        max_extrema = np.array([tmax, zmax])
        min_extrema = np.array([tmin, zmin])

        # Make double sure, that each extremum is significant
        max_dup_idx = np.where(max_extrema[0, 1:] == max_extrema[0, :-1])
        max_extrema = np.delete(max_extrema, max_dup_idx, axis=1)
        min_dup_idx = np.where(min_extrema[0, 1:] == min_extrema[0, :-1])
        min_extrema = np.delete(min_extrema, min_dup_idx, axis=1)

        return max_extrema, min_extrema

    def spline_points(self, T: np.ndarray, extrema: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs spline over given points.

        Parameters
        ----------
        T : numpy array
            Position or time array.
        extrema : numpy array
            Position (1st row) and values (2nd row) of points.

        Returns
        -------
        T : numpy array
            Position array (same as input).
        spline : numpy array
            Spline array over given positions T.
        """

        kind = self.spline_kind.lower()
        t = T[np.r_[T >= extrema[0, 0]] & np.r_[T <= extrema[0, -1]]]

        if kind == "akima":
            return t, akima(extrema[0], extrema[1], t)

        elif kind == "cubic":
            if extrema.shape[1] > 3:
                return t, interp1d(extrema[0], extrema[1], kind=kind)(t)
            else:
                return cubic_spline_3pts(extrema[0], extrema[1], t)

        elif kind in ["slinear", "quadratic", "linear"]:
            return T, interp1d(extrema[0], extrema[1], kind=kind)(t).astype(self.DTYPE)

        else:
            raise ValueError("No such interpolation method!")

    @staticmethod
    def _not_duplicate(S: np.ndarray) -> np.ndarray:
        """
        Returns indices for not repeating values, where there is no extremum.

        Example
        -------
        >>> S = [0, 1, 1, 1, 2, 3]
        >>> idx = self._not_duplicate(S)
        [0, 1, 3, 4, 5]
        """
        dup = np.r_[S[1:-1] == S[0:-2]] & np.r_[S[1:-1] == S[2:]]
        not_dup_idx = np.arange(1, len(S) - 1)[~dup]

        idx = np.empty(len(not_dup_idx) + 2, dtype=np.int64)
        idx[0] = 0
        idx[-1] = len(S) - 1
        idx[1:-1] = not_dup_idx

        return idx

    def find_extrema(self, T: np.ndarray, S: np.ndarray) -> FindExtremaOutput:
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
        if self.extrema_detection == "parabol":
            return self._find_extrema_parabol(T, S)
        elif self.extrema_detection == "simple":
            return self._find_extrema_simple(T, S)
        else:
            raise ValueError("Incorrect extrema detection type. Please try: 'simple' or 'parabol'.")

    def _find_extrema_parabol(self, T: np.ndarray, S: np.ndarray) -> FindExtremaOutput:
        """
        Performs parabol estimation of extremum, i.e. an extremum is a peak
        of parabol spanned on 3 consecutive points, where the mid point is
        the closest.

        See :meth:`EMD.find_extrema()`.
        """
        # Finds indexes of zero-crossings
        S1, S2 = S[:-1], S[1:]
        indzer = np.nonzero(S1 * S2 < 0)[0]
        if np.any(S == 0):
            indz = np.nonzero(S == 0)[0]
            if np.any(np.diff(indz) == 1):
                zer = S == 0
                dz = np.diff(np.append(np.append(0, zer), 0))
                debz = np.nonzero(dz == 1)[0]
                finz = np.nonzero(dz == -1)[0] - 1
                indz = np.round((debz + finz) / 2.0)

            indzer = np.sort(np.append(indzer, indz))

        dt = float(T[1] - T[0])
        scale = 2.0 * dt * dt

        idx = self._not_duplicate(S)
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
        b = (S0 - Sn) * Tp ** 2 + (Sn - Sp) * T0 ** 2 + (Sp - S0) * Tn ** 2
        c = T0 * Tn * T0Tn * Sp + Tn * Tp * TnTp * S0 + Tp * T0 * TpT0 * Sn

        a = a / scale
        b = b / scale
        c = c / scale
        a[a == 0] = 1e-14
        tVertex = -0.5 * b / a
        idx = np.r_[tVertex < T0 + 0.5 * (Tn - T0)] & np.r_[tVertex >= T0 - 0.5 * (T0 - Tp)]

        a, b, c = a[idx], b[idx], c[idx]
        tVertex = tVertex[idx]
        sVertex = a * tVertex * tVertex + b * tVertex + c

        local_max_pos, local_max_val = tVertex[a < 0], sVertex[a < 0]
        local_min_pos, local_min_val = tVertex[a > 0], sVertex[a > 0]

        return local_max_pos, local_max_val, local_min_pos, local_min_val, indzer

    @staticmethod
    def _find_extrema_simple(T: np.ndarray, S: np.ndarray) -> FindExtremaOutput:
        """
        Performs extrema detection, where extremum is defined as a point,
        that is above/below its neighbours.

        See :meth:`EMD.find_extrema`.
        """

        # Finds indexes of zero-crossings
        S1, S2 = S[:-1], S[1:]
        indzer = np.nonzero(S1 * S2 < 0)[0]
        if np.any(S == 0):
            indz = np.nonzero(S == 0)[0]
            if np.any(np.diff(indz) == 1):
                zer = S == 0
                dz = np.diff(np.append(np.append(0, zer), 0))
                debz = np.nonzero(dz == 1)[0]
                finz = np.nonzero(dz == -1)[0] - 1
                indz = np.round((debz + finz) / 2.0)

            indzer = np.sort(np.append(indzer, indz))

        # Finds local extrema
        d = np.diff(S)
        d1, d2 = d[:-1], d[1:]
        indmin = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 < 0])[0] + 1
        indmax = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 > 0])[0] + 1

        # When two or more points have the same value
        if np.any(d == 0):

            imax, imin = [], []

            bad = d == 0
            dd = np.diff(np.append(np.append(0, bad), 0))
            debs = np.nonzero(dd == 1)[0]
            fins = np.nonzero(dd == -1)[0]
            if debs[0] == 1:
                if len(debs) > 1:
                    debs, fins = debs[1:], fins[1:]
                else:
                    debs, fins = [], []

            if len(debs) > 0:
                if fins[-1] == len(S) - 1:
                    if len(debs) > 1:
                        debs, fins = debs[:-1], fins[:-1]
                    else:
                        debs, fins = [], []

            lc = len(debs)
            if lc > 0:
                for k in range(lc):
                    if d[debs[k] - 1] > 0:
                        if d[fins[k]] < 0:
                            imax.append(np.round((fins[k] + debs[k]) / 2.0))
                    else:
                        if d[fins[k]] > 0:
                            imin.append(np.round((fins[k] + debs[k]) / 2.0))

            if len(imax) > 0:
                indmax = indmax.tolist()
                for x in imax:
                    indmax.append(int(x))
                indmax.sort()

            if len(imin) > 0:
                indmin = indmin.tolist()
                for x in imin:
                    indmin.append(int(x))
                indmin.sort()

        local_max_pos = T[indmax]
        local_max_val = S[indmax]
        local_min_pos = T[indmin]
        local_min_val = S[indmin]

        return local_max_pos, local_max_val, local_min_pos, local_min_val, indzer

    def end_condition(self, S: np.ndarray, IMF: np.ndarray) -> bool:
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

        if np.max(tmp) - np.min(tmp) < self.range_thr:
            self.logger.debug("FINISHED -- RANGE")
            return True

        if np.sum(np.abs(tmp)) < self.total_power_thr:
            self.logger.debug("FINISHED -- SUM POWER")
            return True

        return False

    def check_imf(
        self,
        imf_new: np.ndarray,
        imf_old: np.ndarray,
        eMax: np.ndarray,
        eMin: np.ndarray,
    ) -> bool:
        """
        Huang criteria for **IMF** (similar to Cauchy convergence test).
        Signal is an IMF if consecutive siftings do not affect signal
        in a significant manner.
        """
        # local max are >0 and local min are <0
        if np.any(eMax[1] < 0) or np.any(eMin[1] > 0):
            return False

        # Convergence
        if np.sum(imf_new ** 2) < 1e-10:
            return False

        # Precompute values
        imf_diff = imf_new - imf_old
        imf_diff_sqrd_sum = np.sum(imf_diff * imf_diff)

        # Scaled variance test
        svar = imf_diff_sqrd_sum / (max(imf_old) - min(imf_old))
        if svar < self.svar_thr:
            self.logger.debug("Scaled variance -- PASSED")
            return True

        # Standard deviation test
        std = np.sum((imf_diff / imf_new) ** 2)
        if std < self.std_thr:
            self.logger.debug("Standard deviation -- PASSED")
            return True

        energy_ratio = imf_diff_sqrd_sum / np.sum(imf_old * imf_old)
        if energy_ratio < self.energy_ratio_thr:
            self.logger.debug("Energy ratio -- PASSED")
            return True

        return False

    @staticmethod
    def _common_dtype(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Casts inputs (x, y) into a common numpy DTYPE."""
        dtype = np.find_common_type([x.dtype, y.dtype], [])
        if x.dtype != dtype:
            x = x.astype(dtype)
        if y.dtype != dtype:
            y = y.astype(dtype)
        return x, y

    @staticmethod
    def _normalize_time(t: np.ndarray) -> np.ndarray:
        """
        Normalize time array so that it doesn't explode on tiny values.
        Returned array starts with 0 and the smallest increase is by 1.
        """
        d = np.diff(t)
        assert np.all(d != 0), "All time domain values needs to be unique"
        return (t - t[0]) / np.min(d)

    def emd(self, S: np.ndarray, T: Optional[np.ndarray] = None, max_imf: int = -1) -> np.ndarray:
        """
        Performs Empirical Mode Decomposition on signal S.
        The decomposition is limited to *max_imf* imfs.
        Returns IMF functions in numpy array format.

        Parameters
        ----------
        S : numpy array,
            Input signal.
        T : numpy array, (default: None)
            Position or time array. If None is passed or self.extrema_detection == "simple",
            then numpy range is created.
        max_imf : int, (default: -1)
            IMF number to which decomposition should be performed.
            Negative value means *all*.

        Returns
        -------
        IMF : numpy array
            Set of IMFs produced from input signal.
        """
        if T is not None and len(S) != len(T):
            raise ValueError("Time series have different sizes: len(S) -> {} != {} <- len(T)".format(len(S), len(T)))

        if T is None or self.extrema_detection == "simple":
            T = get_timeline(len(S), S.dtype)

        # Normalize T so that it doesn't explode
        T = self._normalize_time(T)

        # Make sure same types are dealt
        S, T = self._common_dtype(S, T)
        self.DTYPE = S.dtype
        N = len(S)

        residue = S.astype(self.DTYPE)
        imf = np.zeros(len(S), dtype=self.DTYPE)
        imf_old = np.nan

        if S.shape != T.shape:
            raise ValueError("Position or time array should be the same size as signal.")

        # Create arrays
        imfNo = 0
        extNo = -1
        IMF = np.empty((imfNo, N))  # Numpy container for IMF
        finished = False

        while not finished:
            self.logger.debug("IMF -- %s", imfNo)

            residue[:] = S - np.sum(IMF[:imfNo], axis=0)
            imf = residue.copy()
            mean = np.zeros(len(S), dtype=self.DTYPE)

            # Counters
            n = 0  # All iterations for current imf.
            n_h = 0  # counts when |#zero - #ext| <=1

            while True:
                n += 1
                if n >= self.MAX_ITERATION:
                    self.logger.info("Max iterations reached for IMF. Continuing with another IMF.")
                    break

                ext_res = self.find_extrema(T, imf)
                max_pos, min_pos, indzer = ext_res[0], ext_res[2], ext_res[4]
                extNo = len(min_pos) + len(max_pos)
                nzm = len(indzer)

                if extNo > 2:

                    max_env, min_env, eMax, eMin = self.extract_max_min_spline(T, imf)
                    mean[:] = 0.5 * (max_env + min_env)

                    imf_old = imf.copy()
                    imf[:] = imf - mean

                    # Fix number of iterations
                    if self.FIXE:
                        if n >= self.FIXE:
                            break

                    # Fix number of iterations after number of zero-crossings
                    # and extrema differ at most by one.
                    elif self.FIXE_H:
                        tmp_residue = self.find_extrema(T, imf)
                        max_pos, min_pos, ind_zer = (
                            tmp_residue[0],
                            tmp_residue[2],
                            tmp_residue[4],
                        )
                        extNo = len(max_pos) + len(min_pos)
                        nzm = len(ind_zer)

                        if n == 1:
                            continue

                        # If proto-IMF add one, or reset counter otherwise
                        n_h = n_h + 1 if abs(extNo - nzm) < 2 else 0

                        # STOP
                        if n_h >= self.FIXE_H:
                            break

                    # Stops after default stopping criteria are met
                    else:
                        ext_res = self.find_extrema(T, imf)
                        max_pos, _, min_pos, _, ind_zer = ext_res
                        extNo = len(max_pos) + len(min_pos)
                        nzm = len(ind_zer)

                        if imf_old is np.nan:
                            continue

                        f1 = self.check_imf(imf, imf_old, eMax, eMin)
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

            if self.end_condition(S, IMF) or imfNo == max_imf:
                finished = True
                break

        # If the last sifting had 2 or less extrema then that's a trend (residue)
        if extNo <= 2:
            IMF = IMF[:-1]

        # Saving imfs and residue for external references
        self.imfs = IMF.copy()
        self.residue = S - np.sum(self.imfs, axis=0)

        # If residue isn't 0 then add it to the output
        if not np.allclose(self.residue, 0):
            IMF = np.vstack((IMF, self.residue))

        return IMF

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and residue from recently analysed signal.

        Returns
        -------
        imfs : np.ndarray
            Obtained IMFs
        residue : np.ndarray
            Residue.

        """
        if self.imfs is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")
        return self.imfs, self.residue

    def get_imfs_and_trend(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and trend from recently analysed signal.
        Note that this may differ from the `get_imfs_and_residue` as the trend isn't
        necessarily the residue. Residue is a point-wise difference between input signal
        and all obtained components, whereas trend is the slowest component (can be zero).

        Returns
        -------
        imfs : np.ndarray
            Obtained IMFs
        trend : np.ndarray
            The main trend.

        """
        if self.imfs is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")

        imfs, residue = self.get_imfs_and_residue()
        if np.allclose(residue, 0):
            return imfs[:-1].copy(), imfs[-1].copy()
        else:
            return imfs, residue


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

    S = np.sin(20 * T * (1 + 0.2 * T)) + T ** 2 + np.sin(13 * T)
    S = S.astype(DTYPE)
    print("Input S.dtype: " + str(S.dtype))

    # Prepare and run EMD
    emd = EMD()
    emd.FIXE_H = 5
    emd.nbsym = 2
    emd.spline_kind = "cubic"
    emd.DTYPE = DTYPE

    imfs = emd.emd(S, T, max_imf)
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
