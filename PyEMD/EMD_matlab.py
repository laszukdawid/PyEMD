#!/usr/bin/python
# coding: UTF-8
#
# Author:   Dawid Laszuk
# Contact:  https://github.com/laszukdawid/PyEMD/issues
#
# Edited:   11/05/2017
#
# Feel free to contact for any information.

import logging
import time

import numpy as np
from scipy.interpolate import interp1d

from PyEMD.splines import akima


class EMD:
    """
    Empirical Mode Decomposition

    *Note:*
    Default and recommended package for EMD is EMD.py.
    This is meant to provide with the same results as MATLAB version of EMD,
    which is not necessarily the most efficient or numerically accurate.

    Method of decomposing signal into Intrinsic Mode Functions (IMFs)
    based on algorithm presented in Huang et al. [1].

    Algorithm was validated with Rilling et al. [2] Matlab's version from 3.2007.

    [1] N. E. Huang et al., "The empirical mode decomposition and the
        Hilbert spectrum for non-linear and non stationary time series
        analysis", Proc. Royal Soc. London A, Vol. 454, pp. 903-995, 1998
    [2] G. Rilling, P. Flandrin and P. Goncalves, "On Empirical Mode
        Decomposition and its algorithms", IEEE-EURASIP Workshop on
        Nonlinear Signal and Image Processing NSIP-03, Grado (I), June 2003
    """

    logger = logging.getLogger(__name__)

    def __init__(self):

        self.splineKind = "cubic"

        self.nbsym = 2
        self.reduceScale = 1.0
        self.maxIteration = 500
        self.scaleFactor = 100

        self.FIXE = 0
        self.FIXE_H = 0

        self.stop1 = 0.05
        self.stop2 = 0.5
        self.stop3 = 0.05

        self.DTYPE = np.float64
        self.MAX_ITERATION = 1000

        self.TIME = False

    def extractMaxMinSpline(self, T, S):
        """
        Input:
        -----------------
            S - Input signal array. Should be 1D.
            T - Time array. If none passed numpy arange is created.

        Output:
        -----------------
            maxSpline - Upper envelope of signal S.
            minSpline - Bottom envelope of signal S.
            maxExtrema - Position (1st row) and values (2nd row) of maxima.
            minExtrema - Position (1st row) and values (2nd row) of minima.
        """

        # Get indexes of extrema
        maxPos, maxVal, minPos, minVal, _ = self.findExtrema(T, S)

        if len(maxPos) + len(minPos) < 3:
            return [-1] * 4

        # Extrapolation of signal (ober boundaries)
        maxExtrema, minExtrema = self.preparePoints(S, T, maxPos, maxVal, minPos, minVal)

        _, maxSpline = self.splinePoints(T, maxExtrema, self.splineKind)
        _, minSpline = self.splinePoints(T, minExtrema, self.splineKind)

        return maxSpline, minSpline, maxExtrema, minExtrema

    def preparePoints(self, S, T, maxPos, maxVal, minPos, minVal):
        """
        Adds to signal extrema according to mirror technique.
        Number of added points depends on nbsym variable.

        Input:
        ---------
            S: Signal (1D numpy array).
            T: Timeline (1D numpy array).
            maxPos: sorted time positions of maxima.
            maxVal: signal values at maxPos positions.
            minPos: sorted time positions of minima.
            minVal: signal values at minPos positions.

        Output:
        ---------
            minExtrema: Position (1st row) and values (2nd row) of minima.
            minExtrema: Position (1st row) and values (2nd row) of maxima.
        """

        # Find indices for time array of extrema
        indmin = np.array([np.nonzero(T == t)[0] for t in minPos]).flatten()
        indmax = np.array([np.nonzero(T == t)[0] for t in maxPos]).flatten()

        # Local variables
        nbsym = self.nbsym
        endMin, endMax = len(minPos), len(maxPos)

        ####################################
        # Left bound - mirror nbsym points to the left
        if indmax[0] < indmin[0]:
            if S[0] > S[indmin[0]]:
                lmax = indmax[1 : min(endMax, nbsym + 1)][::-1]
                lmin = indmin[0 : min(endMin, nbsym + 0)][::-1]
                lsym = indmax[0]
            else:
                lmax = indmax[0 : min(endMax, nbsym)][::-1]
                lmin = np.append(indmin[0 : min(endMin, nbsym - 1)][::-1], 0)
                lsym = 0
        else:
            if S[0] < S[indmax[0]]:
                lmax = indmax[0 : min(endMax, nbsym + 0)][::-1]
                lmin = indmin[1 : min(endMin, nbsym + 1)][::-1]
                lsym = indmin[0]
            else:
                lmax = np.append(indmax[0 : min(endMax, nbsym - 1)][::-1], 0)
                lmin = indmin[0 : min(endMin, nbsym)][::-1]
                lsym = 0

        ####################################
        # Right bound - mirror nbsym points to the right
        if indmax[-1] < indmin[-1]:
            if S[-1] < S[indmax[-1]]:
                rmax = indmax[max(endMax - nbsym, 0) :][::-1]
                rmin = indmin[max(endMin - nbsym - 1, 0) : -1][::-1]
                rsym = indmin[-1]
            else:
                rmax = np.append(indmax[max(endMax - nbsym + 1, 0) :], len(S) - 1)[::-1]
                rmin = indmin[max(endMin - nbsym, 0) :][::-1]
                rsym = len(S) - 1
        else:
            if S[-1] > S[indmin[-1]]:
                rmax = indmax[max(endMax - nbsym - 1, 0) : -1][::-1]
                rmin = indmin[max(endMin - nbsym, 0) :][::-1]
                rsym = indmax[-1]
            else:
                rmax = indmax[max(endMax - nbsym, 0) :][::-1]
                rmin = np.append(indmin[max(endMin - nbsym + 1, 0) :], len(S) - 1)[::-1]
                rsym = len(S) - 1

        # In case any array missing
        if not lmin.size:
            lmin = indmin
        if not rmin.size:
            rmin = indmin
        if not lmax.size:
            lmax = indmax
        if not rmax.size:
            rmax = indmax

        # Mirror points
        tlmin = 2 * T[lsym] - T[lmin]
        tlmax = 2 * T[lsym] - T[lmax]
        trmin = 2 * T[rsym] - T[rmin]
        trmax = 2 * T[rsym] - T[rmax]

        # If mirrored points are not outside passed time range.
        if tlmin[0] > T[0] or tlmax[0] > T[0]:
            if lsym == indmax[0]:
                lmax = indmax[0 : min(endMax, nbsym)][::-1]
            else:
                lmin = indmin[0 : min(endMin, nbsym)][::-1]

            if lsym == 0:
                raise Exception("bug")

            lsym = 0
            tlmin = 2 * T[lsym] - T[lmin]
            tlmax = 2 * T[lsym] - T[lmax]

        if trmin[-1] < T[-1] or trmax[-1] < T[-1]:
            if rsym == indmax[-1]:
                rmax = indmax[max(endMax - nbsym, 0) :][::-1]
            else:
                rmin = indmin[max(endMin - nbsym, 0) :][::-1]

            if rsym == len(S) - 1:
                raise Exception("bug")

            rsym = len(S) - 1
            trmin = 2 * T[rsym] - T[rmin]
            trmax = 2 * T[rsym] - T[rmax]

        zlmax = S[lmax]
        zlmin = S[lmin]
        zrmax = S[rmax]
        zrmin = S[rmin]

        tmin = np.append(tlmin, np.append(T[indmin], trmin))
        tmax = np.append(tlmax, np.append(T[indmax], trmax))
        zmin = np.append(zlmin, np.append(S[indmin], zrmin))
        zmax = np.append(zlmax, np.append(S[indmax], zrmax))

        maxExtrema = np.array([tmax, zmax], dtype=self.DTYPE)
        minExtrema = np.array([tmin, zmin], dtype=self.DTYPE)

        # Make double sure, that each extremum is significant
        maxExtrema = np.delete(maxExtrema, np.where(maxExtrema[0, 1:] == maxExtrema[0, :-1]), axis=1)
        minExtrema = np.delete(minExtrema, np.where(minExtrema[0, 1:] == minExtrema[0, :-1]), axis=1)

        return maxExtrema, minExtrema

    def splinePoints(self, T, extrema, splineKind):
        """
        Constructs spline over given points.

        Input:
        ---------
            T: Time array.
            extrema: Position (1st row) and values (2nd row) of points.
            splineKind: Type of spline.

        Output:
        ---------
            T: Position array.
            spline: Spline over the given points.
        """

        kind = splineKind.lower()
        t = T[np.r_[T >= extrema[0, 0]] & np.r_[T <= extrema[0, -1]]]
        if t.dtype != self.DTYPE:
            self.logger.error("t.dtype: " + str(t.dtype))
        if extrema.dtype != self.DTYPE:
            self.logger.error("extrema.dtype: " + str(extrema.dtype))

        if kind == "akima":
            return t, akima(extrema[0], extrema[1], t)

        elif kind == "cubic":
            if extrema.shape[1] > 3:
                return t, interp1d(extrema[0], extrema[1], kind=kind)(t).astype(self.DTYPE)
            else:
                return self.cubicSpline_3points(T, extrema)

        elif kind in ["slinear", "quadratic", "linear"]:
            return T, interp1d(extrema[0], extrema[1], kind=kind)(t).astype(self.DTYPE)

        else:
            raise ValueError("No such interpolation method!")

    def cubicSpline_3points(self, T, extrema):
        """
        Apparently scipy.interpolate.interp1d does not support
        cubic spline for less than 4 points.
        """

        x0, x1, x2 = extrema[0]
        y0, y1, y2 = extrema[1]

        x1x0, x2x1 = x1 - x0, x2 - x1
        y1y0, y2y1 = y1 - y0, y2 - y1
        _x1x0, _x2x1 = 1.0 / x1x0, 1.0 / x2x1

        m11, m12, m13 = 2 * _x1x0, _x1x0, 0
        m21, m22, m23 = _x1x0, 2.0 * (_x1x0 + _x2x1), _x2x1
        m31, m32, m33 = 0, _x2x1, 2.0 * _x2x1

        v1 = 3 * y1y0 * _x1x0 * _x1x0
        v3 = 3 * y2y1 * _x2x1 * _x2x1
        v2 = v1 + v3

        M = np.matrix([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
        v = np.matrix([v1, v2, v3]).T
        k = np.array(np.linalg.inv(M) * v)

        a1 = k[0] * x1x0 - y1y0
        b1 = -k[1] * x1x0 + y1y0
        a2 = k[1] * x2x1 - y2y1
        b2 = -k[2] * x2x1 + y2y1

        t = T[np.r_[T >= x0] & np.r_[T <= x2]]
        t1 = (T[np.r_[T >= x0] & np.r_[T < x1]] - x0) / x1x0
        t2 = (T[np.r_[T >= x1] & np.r_[T <= x2]] - x1) / x2x1
        t11, t22 = 1.0 - t1, 1.0 - t2

        q1 = t11 * y0 + t1 * y1 + t1 * t11 * (a1 * t11 + b1 * t1)
        q2 = t22 * y1 + t2 * y2 + t2 * t22 * (a2 * t22 + b2 * t2)
        q = np.append(q1, q2)

        return t, q.astype(self.DTYPE)

    @classmethod
    def findExtrema(cls, t, s):
        """
        Finds extrema and zero-crossings.

        Input:
        ---------
            S: Signal.
            T: Time array.

        Output:
        ---------
            localMaxPos: Time positions of maxima.
            localMaxVal: Values of signal at localMaxPos positions.
            localMinPos: Time positions of minima.
            localMinVal: Values of signal at localMinPos positions.
            indzer: Indexes of zero crossings.
        """

        # Finds indexes of zero-crossings
        s1, s2 = s[:-1], s[1:]
        indzer = np.nonzero(s1 * s2 < 0)[0]
        if np.any(s == 0):
            iz = np.nonzero(s == 0)[0]
            indz = []
            if np.any(np.diff(iz) == 1):
                zer = s == 0
                dz = np.diff(np.append(np.append(0, zer), 0))
                debz = np.nonzero(dz == 1)[0]
                finz = np.nonzero(dz == -1)[0] - 1
                indz = np.round((debz + finz) / 2)
            else:
                indz = iz

            indzer = np.sort(np.append(indzer, indz))

        # Finds local extrema
        d = np.diff(s)
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
                if fins[-1] == len(s) - 1:
                    if len(debs) > 1:
                        debs, fins = debs[:-1], fins[:-1]
                    else:
                        debs, fins = [], []

            lc = len(debs)
            if lc > 0:
                for k in range(lc):
                    if d[debs[k] - 1] > 0:
                        if d[fins[k]] < 0:
                            imax.append(round((fins[k] + debs[k]) / 2.0))
                    else:
                        if d[fins[k]] > 0:
                            imin.append(round((fins[k] + debs[k]) / 2.0))

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

        localMaxPos = t[indmax]
        localMaxVal = s[indmax]
        localMinPos = t[indmin]
        localMinVal = s[indmin]

        return localMaxPos, localMaxVal, localMinPos, localMinVal, indzer

    def stop_sifting(self, imf, envMax, envMin, mean, extNo):
        """
        Criterion for stopping sifting process.
        Based on conditions presented in [1].

        [1] G. Rilling, P. Flandrin and P. Goncalves
            "On Empirical Mode Decomposition and its
            algorithms", 2003

        Input:
        ---------
            imf: Current imf.
            envMax: Upper envelope of imf.
            envMin: Bottom envelope of imf.
            mean: Mean of envelopes.
            extNo: Number of extrema.

        Output:
        ---------
            boolean: True if stopping criteria are meet.
        """

        amp = np.abs(envMax - envMin) / 2.0
        sx = np.abs(mean) / amp

        f1 = np.mean(sx > self.stop1) > self.stop3
        f2 = np.any(sx > self.stop2)
        f3 = extNo > 2

        if (not (f1 or f2)) and f3:
            return True
        else:
            return False

    @staticmethod
    def _common_dtype(x, y):

        dtype = np.find_common_type([x.dtype, y.dtype], [])
        if x.dtype != dtype:
            x = x.astype(dtype)
        if y.dtype != dtype:
            y = y.astype(dtype)

        return x, y

    def emd(self, S, T=None, maxImf=None):
        """
        Performs Empirical Mode Decomposition on signal S.
        The decomposition is limited to maxImf imf. No limitation as default.
        Returns IMF functions in dic format. IMF = {0:imf0, 1:imf1...}.

        Input:
        ---------
            S: Signal.
            T: Positions of signal. If none passed numpy arange is created.
            maxImf: IMF number to which decomposition should be performed.
                    As a default, all IMFs are returned.

        Output:
        ---------
        return IMF, EXT, TIME, ITER, imfNo
            IMF: Signal IMFs in dictionary type. IMF = {0:imf0, 1:imf1...}
            EXT: Number of extrema for each IMF. IMF = {0:ext0, 1:ext1...}
            ITER: Number of iteration for each IMF.
            imfNo: Number of IMFs.
        """

        if T is None:
            T = np.arange(len(S), dtype=S.dtype)
        if maxImf is None:
            maxImf = -1

        # Make sure same types are dealt
        S, T = self._common_dtype(S, T)
        self.DTYPE = S.dtype

        Res = S.astype(self.DTYPE)
        scale = 1.0
        Res, scaledS = Res / scale, S / scale
        imf = np.zeros(len(S), dtype=self.DTYPE)
        imfOld = Res.copy()

        if Res.dtype != self.DTYPE:
            self.logger.error("Res.dtype: " + str(Res.dtype))
        if scaledS.dtype != self.DTYPE:
            self.logger.error("scaledS.dtype: " + str(scaledS.dtype))
        if imf.dtype != self.DTYPE:
            self.logger.error("imf.dtype: " + str(imf.dtype))
        if imfOld.dtype != self.DTYPE:
            self.logger.error("imfOld.dtype: " + str(imfOld.dtype))
        if T.dtype != self.DTYPE:
            self.logger.error("T.dtype: " + str(T.dtype))

        if S.shape != T.shape:
            info = "Time array should be the same size as signal."
            raise Exception(info)

        # Create arrays
        IMF = {}  # Dic for imfs signals
        EXT = {}  # Dic for number of extrema
        ITER = {}  # Dic for number of iterations
        TIME = {}  # Dic for time of computation
        imfNo = 0
        extNo = 0
        notFinish = True

        while notFinish:
            self.logger.debug("IMF -- " + str(imfNo))

            # ~ Res = scaledS - np.sum([IMF[i] for i in range(imfNo)],axis=0)
            Res -= imf
            imf = Res.copy()
            mean = np.zeros(len(S), dtype=self.DTYPE)

            # Counters
            n = 0  # All iterations for current imf.
            n_h = 0  # counts when |#zero - #ext| <=1

            # Time counter
            timeInit = time.time()
            if self.TIME:
                singleTime = time.time()

            while n < self.MAX_ITERATION:
                n += 1

                if self.TIME:
                    self.logger.info("Execution time: " + str(time.time() - singleTime))
                    singleTime = time.time()
                ext_res = self.findExtrema(T, imf)
                MP, mP = ext_res[0], ext_res[2]
                indzer = ext_res[4]

                extNo = len(mP) + len(MP)
                nzm = len(indzer)

                if extNo > 2:

                    # Plotting. Either into file, or on-screen display.
                    imfOld = imf.copy()
                    imf = imf - self.reduceScale * mean

                    env_ext = self.extractMaxMinSpline(T, imf)
                    maxEnv, minEnv = env_ext[0], env_ext[1]

                    if isinstance(maxEnv, int):
                        notFinish = True
                        break

                    mean = 0.5 * (maxEnv + minEnv)

                    if maxEnv.dtype != self.DTYPE:
                        self.logger.error("maxEnv.dtype: " + str(maxEnv.dtype))
                    if minEnv.dtype != self.DTYPE:
                        self.logger.error("minEnv.dtype: " + str(minEnv.dtype))
                    if imf.dtype != self.DTYPE:
                        self.logger.error("imf.dtype: " + str(imf.dtype))
                    if mean.dtype != self.DTYPE:
                        self.logger.error("mean.dtype: " + str(mean.dtype))

                    # Stop, because of too many iterations
                    if n > self.maxIteration:
                        self.logger.info("TOO MANY ITERATIONS! BREAK!")
                        break

                    # Fix number of iterations
                    if self.FIXE:
                        if n >= self.FIXE + 1:
                            break

                    # Fix number of iterations after number of zero-crossings
                    # and extrema differ at most by one.
                    elif self.FIXE_H:

                        ext_res = self.findExtrema(T, imf)
                        mP, MP, indzer = ext_res[0], ext_res[2], ext_res[4]
                        extNo = len(mP) + len(MP)
                        nzm = len(indzer)

                        if n == 1:
                            continue
                        if abs(extNo - nzm) > 1:
                            n_h = 0
                        else:
                            n_h += 1

                        # STOP
                        if n_h >= self.FIXE_H:
                            break

                    # Stops after default stopping criteria are meet.
                    else:

                        mP, _, MP, _, indzer = self.findExtrema(T, imf)
                        extNo = len(mP) + len(MP)
                        nzm = len(indzer)

                        f1 = self.stop_sifting(imf, maxEnv, minEnv, mean, extNo)
                        f2 = abs(extNo - nzm) < 2

                        # STOP
                        if f1 and f2:
                            break

                else:
                    notFinish = False
                    break

            IMF[imfNo] = imf.copy()
            ITER[imfNo] = n
            EXT[imfNo] = extNo
            TIME[imfNo] = time.time() - timeInit
            imfNo += 1

            if imfNo == maxImf - 1:
                notFinish = False
                break

        # ~ Saving residuum if meaningful
        Res = scaledS - np.sum([IMF[i] for i in range(imfNo)], axis=0)
        if np.sum(np.abs(Res)) > 1e-10:
            IMF[imfNo] = Res
            ITER[imfNo] = 0
            EXT[imfNo] = extNo
            TIME[imfNo] = 0
            imfNo += 1

        for key in list(IMF.keys()):
            IMF[key] *= scale
        return IMF, EXT, ITER, imfNo


###################################################
# Beginning of program

if __name__ == "__main__":

    import pylab as plt

    # Logging options
    logging.basicConfig(level=logging.DEBUG)

    # EMD options
    maxImf = -1
    DTYPE = np.float64

    # Signal options
    N = 1000
    tMin, tMax = 0, 1
    T = np.linspace(tMin, tMax, N, dtype=DTYPE)

    S = 6 * T + np.cos(8 * np.pi ** T) + 0.5 * np.cos(40 * np.pi * T)
    S = S.astype(DTYPE)

    # Prepare and run EMD
    emd = EMD()
    emd.FIXE_H = 5
    # ~ emd.FIXE = 10
    emd.nbsym = 2
    emd.splineKind = "cubic"
    emd.DTYPE = DTYPE
    IMF, EXT, ITER, imfNo = emd.emd(S, T, maxImf)

    # Save results (IMFs) into file
    npIMF = np.zeros((imfNo, N), dtype=DTYPE)
    for i in range(imfNo):
        npIMF[i] = IMF[i]

    np.save("imfs", npIMF)

    # Plotting
    # ~ c = np.floor(np.sqrt(imfNo+3))
    # ~ r = np.ceil( (imfNo+3)/c)
    c = np.floor(np.sqrt(imfNo + 1))
    r = np.ceil((imfNo + 1) / c)

    plt.ioff()
    plt.subplot(r, c, 1)
    plt.plot(T, S, "r")
    plt.title("Original signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    # ~ plt.subplot(r,c,2)
    # ~ plt.plot([EXT[i] for i in range(imfNo)], 'o')
    # ~ plt.ylim(0, max([EXT[i] for i in range(imfNo)])+1)
    # ~ plt.title("Number of extrema")
    # ~
    # ~ plt.subplot(r,c,3)
    # ~ plt.plot([ITER[i] for i in range(imfNo)], 'o')
    # ~ plt.ylim(0, max([ITER[i] for i in range(imfNo)])+1)
    # ~ plt.title("Number of iterations")

    for num in range(imfNo):
        # ~ plt.subplot(r,c,num+4)
        plt.subplot(r, c, num + 2)
        plt.plot(T, IMF[num], "g")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

        if num == imfNo - 1:
            plt.title("Residue")
        else:
            plt.title("Imf " + str(num))

    plt.tight_layout()
    plt.show()
