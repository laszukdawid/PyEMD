#!/usr/bin/python
# coding: UTF-8
#
# Author:   Dawid Laszuk
# Contact:  laszukdawid@gmail.com
#
# Edited:   11/05/2017
#
# Feel free to contact for any information.

from __future__ import division, print_function

import logging
import numpy as np
import os

from scipy.interpolate import interp1d
from PyEMD.splines import *

class EMD:
    """
    Empirical Mode Decomposition

    *Note:*
    Default and recommended package for EMD is EMD.py.
    Advantage of this is that it utilises Kahan summation algorithm,
    which reduces numerical error when subtracting numerically.

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
        # Declare constants
        self.stdThreshold = 0.2
        self.scaledVarThreshold = 0.001
        self.powerThreshold = -5
        self.totalPowerThreshold = 0.01
        self.rangeThreshold = 0.001

        self.nbsym = 2
        self.reduceScale = 1.
        self.scaleFactor = 100.

        self.PLOT = 0
        self.INTERACTIVE = 0
        self.plotPath = 'splineTest'

        self.splineKind = 'akima'
        self.extrema_detection = 'simple' # simple, parabol

        self.DTYPE = np.float64
        self.FIXE = 0
        self.FIXE_H = 0

        self.MAX_ITERATION = 1000

        if self.PLOT:
            import pylab as plt

    def extract_max_min_spline(self, T, S):
        """
        Input:
        -----------------
            T - Time array.
            S - Signal.

        Output:
        -----------------
            maxSpline - Spline which connects maxima of S.
            minSpline - Spline which connects minima of S.
        """

        # Get indexes of extrema
        maxPos, maxVal, minPos, minVal, indzer = self.find_extrema(T, S)

        if maxPos.dtype!=self.DTYPE: self.logger.error('maxPos.dtype: '+str(maxPos.dtype))
        if maxVal.dtype!=self.DTYPE: self.logger.error('maxVal.dtype: '+str(maxVal.dtype))
        if minPos.dtype!=self.DTYPE: self.logger.error('minPos.dtype: '+str(minPos.dtype))
        if minVal.dtype!=self.DTYPE: self.logger.error('minVal.dtype: '+str(minVal.dtype))
        if len(maxPos) + len(minPos) < 3: return [-1]*4

        #########################################
        # Extrapolation of signal (over boundaries)
        maxExtrema, minExtrema = self.prepare_points(T, S, maxPos, maxVal, minPos, minVal)

        maxTSpline, maxSpline = self.spline_points(T, maxExtrema)
        minTSpline, minSpline = self.spline_points(T, minExtrema)

        if maxExtrema.dtype!=self.DTYPE: self.logger.error('maxExtrema.dtype: '+str(maxExtrema.dtype))
        if maxSpline.dtype!=self.DTYPE: self.logger.error('maxSpline.dtype: '+str(maxSpline.dtype))
        if maxTSpline.dtype!=self.DTYPE: self.logger.error('maxTSline.dtype: '+str(maxTSpline.dtype))

        return maxSpline, minSpline, maxExtrema, minExtrema

    def prepare_points(self, T, S, maxPos, maxVal, minPos, minVal):
        if self.extrema_detection=="parabol":
            return self._prepare_points_parabol(T, S, maxPos, maxVal, minPos, minVal)
        elif self.extrema_detection=="simple":
            return self._prepare_points_simple(T, S, maxPos, maxVal, minPos, minVal)
        else:
            msg = "Incorrect extrema detection type. Please try: "
            msg+= "'simple' or 'parabol'."
            raise ValueError(msg)

    def _prepare_points_parabol(self, T, S, maxPos, maxVal, minPos, minVal):
        """
        Input:
        ---------
            S - Signal values (1D numpy array).
            T - Timeline of values (1D numpy array).
            extrema - Indexes of extrema points (1D list).

        Output:
        ---------
            leftP - (time, value) of left mirrored extrema.
            rightP - (time, value) of right mirrored extrema.
        """

        # Need at least two extrema to perform mirroring
        maxExtrema = np.zeros((2,len(maxPos)), dtype=self.DTYPE)
        minExtrema = np.zeros((2,len(minPos)), dtype=self.DTYPE)

        maxExtrema[0], minExtrema[0] = maxPos, minPos
        maxExtrema[1], minExtrema[1] = maxVal, minVal

        # Local variables
        nbsym = self.nbsym
        endMin, endMax = len(minPos), len(maxPos)

        ####################################
        # Left bound
        dPos = maxPos[0] - minPos[0]
        leftExtType = ["min", "max"][dPos<0]

        if (leftExtType == "max"):
            if (S[0]>minVal[0]) and (np.abs(dPos)>(maxPos[0]-T[0])):
                # mirror signal to first extrema
                expandLeftMaxPos = 2*maxPos[0] - maxPos[1:nbsym+1]
                expandLeftMinPos = 2*maxPos[0] - minPos[0:nbsym]
                expandLeftMaxVal = maxVal[1:nbsym+1]
                expandLeftMinVal = minVal[0:nbsym]

            else:
                # mirror signal to beginning
                expandLeftMaxPos = 2*T[0] - maxPos[0:nbsym]
                expandLeftMinPos = 2*T[0] - np.append(T[0], minPos[0:nbsym-1])
                expandLeftMaxVal = maxVal[0:nbsym]
                expandLeftMinVal = np.append(S[0], minVal[0:nbsym-1])


        elif (leftExtType == "min"):
            if (S[0] < maxVal[0]) and (np.abs(dPos)>(minPos[0]-T[0])):
                # mirror signal to first extrema
                expandLeftMaxPos = 2*minPos[0] - maxPos[0:nbsym]
                expandLeftMinPos = 2*minPos[0] - minPos[1:nbsym+1]
                expandLeftMaxVal = maxVal[0:nbsym]
                expandLeftMinVal = minVal[1:nbsym+1]

            else:
                # mirror signal to beginning
                expandLeftMaxPos = 2*T[0] - np.append(T[0], maxPos[0:nbsym-1])
                expandLeftMinPos = 2*T[0] - minPos[0:nbsym]
                expandLeftMaxVal = np.append(S[0], maxVal[0:nbsym-1])
                expandLeftMinVal = minVal[0:nbsym]

        if not expandLeftMinPos.shape:
            expandLeftMinPos, expandLeftMinVal = minPos, minVal
        if not expandLeftMaxPos.shape:
            expandLeftMaxPos, expandLeftMaxVal = maxPos, maxVal

        expandLeftMin = np.vstack((expandLeftMinPos[::-1], expandLeftMinVal[::-1]))
        expandLeftMax = np.vstack((expandLeftMaxPos[::-1], expandLeftMaxVal[::-1]))

        ####################################
        # Right bound
        dPos = maxPos[-1] - minPos[-1]
        rightExtType = ["min","max"][dPos>0]

        if (rightExtType == "min"):
            if (S[-1] < maxVal[-1]) and (np.abs(dPos)>(T[-1]-minPos[-1])):
                # mirror signal to last extrema
                idxMax = max(0, endMax-nbsym)
                idxMin = max(0, endMin-nbsym-1)
                expandRightMaxPos = 2*minPos[-1] - maxPos[idxMax:]
                expandRightMinPos = 2*minPos[-1] - minPos[idxMin:-1]
                expandRightMaxVal = maxVal[idxMax:]
                expandRightMinVal = minVal[idxMin:-1]
            else:
                # mirror signal to end
                idxMax = max(0, endMax-nbsym+1)
                idxMin = max(0, endMin-nbsym)
                expandRightMaxPos = 2*T[-1] - np.append(maxPos[idxMax:], T[-1])
                expandRightMinPos = 2*T[-1] - minPos[idxMin:]
                expandRightMaxVal = np.append(maxVal[idxMax:],S[-1])
                expandRightMinVal = minVal[idxMin:]

        elif (rightExtType == "max"):
            if (S[-1] > minVal[-1]) and len(maxPos)>1 and (np.abs(dPos)>(T[-1]-maxPos[-1])):
                # mirror signal to last extremum
                idxMax = max(0, endMax-nbsym-1)
                idxMin = max(0, endMin-nbsym)
                expandRightMaxPos = 2*maxPos[-1] - maxPos[idxMax:-1]
                expandRightMinPos = 2*maxPos[-1] - minPos[idxMin:]
                expandRightMaxVal = maxVal[idxMax:-1]
                expandRightMinVal = minVal[idxMin:]
            else:
                # mirror signal to end
                idxMax = max(0, endMax-nbsym)
                idxMin = max(0, endMin-nbsym+1)
                expandRightMaxPos = 2*T[-1] - maxPos[idxMax:]
                expandRightMinPos = 2*T[-1] - np.append(minPos[idxMin:], T[-1])
                expandRightMaxVal = maxVal[idxMax:]
                expandRightMinVal = np.append(minVal[idxMin:], S[-1])



        if not expandRightMinPos.shape:
            expandRightMinPos, expandRightMinVal = minPos, minVal
        if not expandRightMaxPos.shape:
            expandRightMaxPos, expandRightMaxVal = maxPos, maxVal

        expandRightMin = np.vstack((expandRightMinPos[::-1], expandRightMinVal[::-1]))
        expandRightMax = np.vstack((expandRightMaxPos[::-1], expandRightMaxVal[::-1]))

        maxExtrema = np.hstack((expandLeftMax, maxExtrema, expandRightMax))
        minExtrema = np.hstack((expandLeftMin, minExtrema, expandRightMin))

        return maxExtrema, minExtrema

    def _prepare_points_simple(self, T, S, maxPos, maxVal, minPos, minVal):
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

        # Find indexes of pass
        indmin = np.array([np.nonzero(T==t)[0] for t in minPos]).flatten()
        indmax = np.array([np.nonzero(T==t)[0] for t in maxPos]).flatten()

        if S.dtype != self.DTYPE: self.logger.error('S.dtype: '+str(S.dtype))
        if T.dtype != self.DTYPE: self.logger.error('T.dtype: '+str(T.dtype))

        # Local variables
        nbsym = self.nbsym
        endMin, endMax = len(minPos), len(maxPos)

        ####################################
        # Left bound - mirror nbsym points to the left
        if indmax[0] < indmin[0]:
            if S[0] > S[indmin[0]]:
                lmax = indmax[1:min(endMax,nbsym+1)][::-1]
                lmin = indmin[0:min(endMin,nbsym+0)][::-1]
                lsym = indmax[0]
            else:
                lmax = indmax[0:min(endMax,nbsym)][::-1]
                lmin = np.append(indmin[0:min(endMin,nbsym-1)][::-1],0)
                lsym = 0
        else:
            if S[0] < S[indmax[0]]:
                lmax = indmax[0:min(endMax,nbsym+0)][::-1]
                lmin = indmin[1:min(endMin,nbsym+1)][::-1]
                lsym = indmin[0]
            else:
                lmax = np.append(indmax[0:min(endMax,nbsym-1)][::-1],0)
                lmin = indmin[0:min(endMin,nbsym)][::-1]
                lsym = 0

        ####################################
        # Right bound - mirror nbsym points to the right
        if indmax[-1] < indmin[-1]:
            if S[-1] < S[indmax[-1]]:
                rmax = indmax[max(endMax-nbsym,0):][::-1]
                rmin = indmin[max(endMin-nbsym-1,0):-1][::-1]
                rsym = indmin[-1]
            else:
                rmax = np.append(indmax[max(endMax-nbsym+1,0):], len(S)-1)[::-1]
                rmin = indmin[max(endMin-nbsym,0):][::-1]
                rsym = len(S)-1
        else:
            if S[-1] > S[indmin[-1]]:
                rmax = indmax[max(endMax-nbsym-1,0):-1][::-1]
                rmin = indmin[max(endMin-nbsym,0):][::-1]
                rsym = indmax[-1]
            else:
                rmax = indmax[max(endMax-nbsym,0):][::-1]
                rmin = np.append(indmin[max(endMin-nbsym+1,0):], len(S)-1)[::-1]
                rsym = len(S)-1

        # In case any array missing
        if not lmin.size: lmin = indmin
        if not rmin.size: rmin = indmin
        if not lmax.size: lmax = indmax
        if not rmax.size: rmax = indmax

        # Mirror points
        tlmin = 2*T[lsym]-T[lmin]
        tlmax = 2*T[lsym]-T[lmax]
        trmin = 2*T[rsym]-T[rmin]
        trmax = 2*T[rsym]-T[rmax]

        # If mirrored points are not outside passed time range.
        if tlmin[0] > T[0] or tlmax[0] > T[0]:
            if lsym == indmax[0]:
                lmax = indmax[0:min(endMax,nbsym)][::-1]
            else:
                lmin = indmin[0:min(endMin,nbsym)][::-1]

            if lsym == 0:
                raise Exception('Left edge BUG')

            lsym = 0
            tlmin = 2*T[lsym]-T[lmin]
            tlmax = 2*T[lsym]-T[lmax]

        if trmin[-1] < T[-1] or trmax[-1] < T[-1]:
            if rsym == indmax[-1]:
                rmax = indmax[max(endMax-nbsym,0):][::-1]
            else:
                rmin = indmin[max(endMin-nbsym,0):][::-1]

            if rsym == len(S)-1:
                raise Exception('Right edge BUG')

            rsym = len(S)-1
            trmin = 2*T[rsym]-T[rmin]
            trmax = 2*T[rsym]-T[rmax]

        zlmax = S[lmax]
        zlmin = S[lmin]
        zrmax = S[rmax]
        zrmin = S[rmin]

        tmin = np.append(tlmin, np.append(T[indmin], trmin))
        tmax = np.append(tlmax, np.append(T[indmax], trmax))
        zmin = np.append(zlmin, np.append(S[indmin], zrmin))
        zmax = np.append(zlmax, np.append(S[indmax], zrmax))

        maxExtrema = np.array([tmax, zmax])
        minExtrema = np.array([tmin, zmin])
        if maxExtrema.dtype != self.DTYPE: self.logger.error('maxExtrema.dtype: '+str(maxExtrema.dtype))

        # Make double sure, that each extremum is significant
        maxExtrema = np.delete(maxExtrema, np.where(maxExtrema[0,1:]==maxExtrema[0,:-1]),axis=1)
        minExtrema = np.delete(minExtrema, np.where(minExtrema[0,1:]==minExtrema[0,:-1]),axis=1)

        return maxExtrema, minExtrema

    def spline_points(self, T, extrema):
        """
        Constructs spline over given points.

        Input:
        ---------
            T: Time array.
            extrema: Position (1st row) and values (2nd row) of points.
            splineKind: Type of spline.

        Output:
        ---------
            T: Poistion array.
            spline: Spline over the given points.
        """

        kind = self.splineKind.lower()
        t = T[np.r_[T>=extrema[0,0]] & np.r_[T<=extrema[0,-1]]]
        if t.dtype != self.DTYPE: self.logger.error('t.dtype: '+str(t.dtype))
        if extrema.dtype != self.DTYPE: self.logger.error('extrema.dtype: '+str(extrema.dtype))

        if kind == "akima":
            return t, akima(extrema[0], extrema[1], t)

        elif kind == 'cubic':
            if extrema.shape[1]>3:
                return t, interp1d(extrema[0], extrema[1], kind=kind)(t)
            else:
                return cubic_spline_3pts(extrema[0], extrema[1], t)

        elif kind in ['slinear', 'quadratic', 'linear']:
            return T, interp1d(extrema[0], extrema[1], kind=kind)(t).astype(self.DTYPE)

        else:
            raise ValueError("No such interpolation method!")

    def not_duplicate(self, s):
        idx = [0]
        for i in range(1,len(s)-1):
            if (s[i] == s[i+1] and s[i] == s[i-1]):
               pass

            else: idx.append(i)
        idx.append(len(s)-1)
        return idx

    def find_extrema(self, t, s):
        if self.extrema_detection=="parabol":
            return self._find_extrema_parabol(t, s)
        elif self.extrema_detection=="simple":
            return self._find_extrema_simple(t, s)
        else:
            msg = "Incorrect extrema detection type. Please try: "
            msg+= "'simple' or 'parabol'."
            raise ValueError(msg)

    def _find_extrema_parabol(self, t, s):
        """
        Estimates position and value of extrema by parabolic
        interpolation based on three consecutive points.

        Input:
        ------------
            t - time array;
            s - signal;

        Output:
        ------------
            localMaxPos - position of local maxima;
            localMaxVal - values of local maxima;
            localMinPos - position of local minima;
            localMinVal - values of local minima;

        """
        # Finds indexes of zero-crossings
        s1, s2 = s[:-1], s[1:]
        indzer = np.nonzero(s1*s2<0)[0]
        if np.any(s == 0):
            iz = np.nonzero(s==0)[0]
            indz = []
            if np.any(np.diff(iz)==1):
                zer = s == 0
                dz = np.diff(np.append(np.append(0, zer), 0))
                debz = np.nonzero(dz == 1)[0]
                finz = np.nonzero(dz == -1)[0]-1
                indz = np.round((debz+finz)/2.)
            else:
                indz = iz

            indzer = np.sort(np.append(indzer, indz))

        dt = float(t[1]-t[0])
        scale = 2.*dt*dt

        idx = self.not_duplicate(s)
        t = t[idx]
        s = s[idx]

        # p - previous
        # 0 - current
        # n - next
        tp, t0, tn = t[:-2], t[1:-1], t[2:]
        sp, s0, sn = s[:-2], s[1:-1], s[2:]
        #~ a = sn + sp - 2*s0
        #~ b = 2*(tn+tp)*s0 - ((tn+t0)*sp+(t0+tp)*sn)
        #~ c = sp*t0*tn -2*tp*s0*tn + tp*t0*sn
        tntp, t0tn, tpt0 = tn-tp, t0-tn, tp-t0
        #~ scale = tp*tn*tn + tp*tp*t0 + t0*t0*tn - (tp*tp*tn + tp*t0*t0 + t0*tn*tn)
        scale = -tntp*tpt0*t0tn

        #~ a = t0tn*sp + tntp*s0 + tpt0*sn
        #~ b = (s0-sn)*tp**2 + (sn-sp)*t0**2 + (sp-s0)*tn**2
        #~ c = t0*tn*t0tn*sp + tn*tp*tntp*s0 + tp*t0*tpt0*sn
        #~ a = tn*(s0-sp) + t0*(sp-sn) + tp*(sn-s0)
        a = tn*s0 + t0*sp + tp*sn - (tn*sp + t0*sn + tp*s0)
        b = (s0-sn)*tp**2 + (sn-sp)*t0**2 + (sp-s0)*tn**2
        c = t0*tn*t0tn*sp + tn*tp*tntp*s0 + tp*t0*tpt0*sn

        a = a/scale
        b = b/scale
        c = c/scale

        tVertex = np.zeros(a.size, dtype=self.DTYPE)

        # Extremum only if a!=0
        nonZeroA = a!=0
        tVertex[nonZeroA] = -0.5*b[nonZeroA]/a[nonZeroA]

        # If a==0 than it's a straight line, thus no extremum
        zeroA = a==0
        tVertex[zeroA] = tn[zeroA]

        idx = np.r_[tVertex<(t0 + 0.5*(tn-t0))] & np.r_[tVertex>=(t0-0.5*(t0-tp))]

        I = []
        for i in np.arange(len(idx))[idx]:#[:-1]:
            if i > 2 and (i < len(t0)-2):
                if  sp[i-1] >= s0[i-1] and sp[i] >= s0[i] and s0[i] >= sn[i] and s0[i+1] >= sn[i+1]:
                    pass
                elif sp[i-1] <= s0[i-1] and sp[i] <= s0[i] and  s0[i] <= sn[i] and s0[i+1] <= sn[i+1]:
                    pass
                else:
                    I.append(i)
            else:
                I.append(i)

        idx = np.array(I)
        a, b, c = a[idx], b[idx], c[idx]

        tVertex = tVertex[idx]
        T, S = t0[idx], s0[idx]
        #~ sVertex = a*(tVertex+T)*(tVertex-T) + b*(tVertex-T) + S
        sVertex = a*tVertex*tVertex + b*tVertex + c

        localMaxPos, localMaxVal = tVertex[a<0], sVertex[a<0]
        localMinPos, localMinVal = tVertex[a>0], sVertex[a>0]

        return localMaxPos, localMaxVal, localMinPos, localMinVal, indzer

    def _find_extrema_simple(self, t, s):
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
        indzer = np.nonzero(s1*s2<0)[0]
        if np.any(s == 0):
            iz = np.nonzero( s==0 )[0]
            indz = []
            if np.any(np.diff(iz)==1):
                zer = s == 0
                dz = np.diff(np.append(np.append(0, zer), 0))
                debz = np.nonzero(dz == 1)[0]
                finz = np.nonzero(dz == -1)[0]-1
                indz = np.round((debz+finz)/2.)
            else:
                indz = iz

            indzer = np.sort(np.append(indzer, indz))


        # Finds local extrema
        d = np.diff(s)
        d1, d2 = d[:-1], d[1:]
        indmin = np.nonzero(np.r_[d1*d2<0] & np.r_[d1<0])[0]+1
        indmax = np.nonzero(np.r_[d1*d2<0] & np.r_[d1>0])[0]+1

        # When two or more points have the same value
        if np.any(d==0):

            imax, imin = [], []

            bad = (d==0)
            dd = np.diff(np.append(np.append(0, bad), 0))
            debs = np.nonzero(dd == 1)[0]
            fins = np.nonzero(dd == -1)[0]
            if debs[0] == 1:
                if len(debs) > 1:
                    debs, fins = debs[1:], fins[1:]
                else:
                    debs, fins = [], []

            if len(debs) > 0:
                if fins[-1] == len(s)-1:
                    if len(debs) > 1:
                        debs, fins = debs[:-1], fins[:-1]
                    else:
                        debs, fins = [], []

            lc = len(debs)
            if lc > 0:
                for k in range(lc):
                    if d[debs[k]-1] > 0:
                        if d[fins[k]] < 0:
                            imax.append(np.round((fins[k]+debs[k])/2.))
                    else:
                        if d[fins[k]] > 0:
                            imin.append(np.round((fins[k]+debs[k])/2.))

            if len(imax) > 0:
                indmax = indmax.tolist()
                for x in imax: indmax.append(int(x))
                indmax.sort()

            if len(imin) > 0:
                indmin = indmin.tolist()
                for x in imin: indmin.append(int(x))
                indmin.sort()

        localMaxPos = t[indmax]
        localMaxVal = s[indmax]
        localMinPos = t[indmin]
        localMinVal = s[indmin]

        return localMaxPos, localMaxVal, localMinPos, localMinVal, indzer

    def findExtrema(self, t, s):
        """
        Estimates position and value of extrema by parabolic
        interpolation based on three consecutive points.

        Input:
        ------------
            t - time array;
            s - signal;

        Output:
        ------------
            localMaxPos - position of local maxima;
            localMaxVal - values of local maxima;
            localMinPos - position of local minima;
            localMinVal - values of local minima;

        """
        # Finds indexes of zero-crossings
        s1, s2 = s[:-1], s[1:]
        indzer = np.nonzero(s1*s2<0)[0]
        if np.any(s == 0):
            iz = np.nonzero(s==0)[0]
            indz = []
            if np.any(np.diff(iz)==1):
                zer = s == 0
                dz = np.diff(np.append(np.append(0, zer), 0))
                debz = np.nonzero(dz == 1)[0]
                finz = np.nonzero(dz == -1)[0]-1
                indz = np.round((debz+finz)/2.)
            else:
                indz = iz

            indzer = np.sort(np.append(indzer, indz))


        d = np.append(np.diff(s), 1)
        t = t[d!=0]
        s = s[d!=0]

        dt = t[1]-t[0]
        tVertex = np.zeros(len(t)-2, dtype=self.DTYPE)

        # p - previous
        # 0 - current
        # n - next
        tp, t0, tn = t[:-2], t[1:-1], t[2:]
        sp, s0, sn = s[:-2], s[1:-1], s[2:]
        a = sn + sp - 2*s0
        #~ b = 2*(tn+tp)*s0 - ((tn+t0)*sp+(t0+tp)*sn)
        b = dt*(sn-sp) + 2*t0*(2*s0 - (sn+sp))


        # Vertex positions
        idx = a!=0
        tVertex[idx] = -0.5*b[idx]/a[idx]


        # Extract only vertices in considered range
        idx = np.r_[tVertex<=tn-0.5*dt] & np.r_[tVertex>tp+0.5*dt]
        a, b = a[idx], b[idx]
        a, b = a/(2.*dt*dt), b/(2.*dt*dt)

        tVertex = tVertex[idx]
        T, S = t0[idx], s0[idx]

        # Estimates value of vertex - c = S - a T^2 - b T
        sVertex = a*(tVertex+T)*(tVertex-T) + b*(tVertex-T) + S
        #~ sVertex = a*tVertex*tVertex + b*tVertex + c

        localMaxPos, localMaxVal = tVertex[a<0], sVertex[a<0]
        localMinPos, localMinVal = tVertex[a>0], sVertex[a>0]

        return localMaxPos, localMaxVal, localMinPos, localMinVal, indzer


    def end_condition(self, Res, IMF):
        # When to stop EMD
        tmp = Res.copy()
        for imfNo in list(IMF.keys()):
            tmp -= IMF[imfNo]

        #~ # Power is enough
        #~ if np.log10(np.abs(tmp).sum()/np.abs(Res).sum()) < powerThreshold:
            #~ self.logger.info("FINISHED -- POWER RATIO")
            #~ return True

        if np.max(tmp) - np.min(tmp) < self.rangeThreshold:
            self.logger.info("FINISHED -- RANGE")
            return True

        if np.sum(np.abs(tmp)) < self.totalPowerThreshold:
            self.logger.info("FINISHED -- SUM POWER")
            return True

    def check_imf(self, imfNew, imfOld, eMax, eMin, mean):
        """
        Huang criteria. Similar to Cauchy convergence test.
        SD stands for Sum of the Difference.
        """
        # local max are >0 and local min are <0
        if np.any(eMax[1]<0) or np.any(eMin[1]>0):
            return False

        # Convergence
        if np.sum(imfNew**2) < 1e-10: return False

        std = np.sum( ((imfNew-imfOld)/imfNew)**2 )
        scaledVar = np.sum((imfNew-imfOld)**2)/(max(imfOld)-min(imfOld))


        if  scaledVar < self.scaledVarThreshold:
            self.logger.info("Scaled variance -- PASSED")
            return True
        elif std < self.stdThreshold:
            self.logger.info("Standard deviation -- PASSED")
            return True
        else:
            return False

    def _common_dtype(self, x, y):

        dtype = np.find_common_type([x.dtype, y.dtype], [])
        if x.dtype != dtype: x = x.astype(dtype)
        if y.dtype != dtype: y = y.astype(dtype)

        return x, y

    def emd(self, S, T=None, maxImf=None):
        """
        Performs Empirical Mode Decomposition on signal S.
        The decomposition is limited to maxImf imf. No limitation as default.
        Returns IMF functions in numpy array format.

        Input:
        ---------
            S: Signal.
            T: Positions of signal. If none passed numpy arange is created.
            maxImf: IMF number to which decomposition should be performed.
                    As a default, all IMFs are returned.

        Output:
        ---------
            IMF: Signal IMFs in numpy array form.
        """

        if T is None: T = np.arange(len(S), dtype=S.dtype)
        if maxImf is None: maxImf = -1

        # Make sure same types are dealt
        S, T = self._common_dtype(S, T)
        self.DTYPE = S.dtype

        Res = S.astype(self.DTYPE)
        scale = (max(Res) - min(Res))/float(self.scaleFactor)
        Res, scaledS = Res/scale, S/scale
        imf = np.zeros(len(S), dtype=self.DTYPE)
        imfOld = Res.copy()

        N = len(S)

        if Res.dtype!=self.DTYPE: self.logger.error('Res.dtype: '+str(Res.dtype))
        if scaledS.dtype!=self.DTYPE: self.logger.error('scaledS.dtype: '+str(scaledS.dtype))
        if imf.dtype!=self.DTYPE: self.logger.error('imf.dtype: '+str(imf.dtype))
        if imfOld.dtype!=self.DTYPE: self.logger.error('imfOld.dtype: '+str(imfOld.dtype))
        if T.dtype!=self.DTYPE: self.logger.error('T.dtype: '+str(T.dtype))

        if S.shape != T.shape:
            info = "Time array should be the same size as signal."
            raise ValueError(info)

        # Create arrays
        IMF = {} # Dict for imfs signals
        imfNo = 0
        notFinish = True

        corRes = np.zeros(N, dtype=self.DTYPE)
        y = np.zeros(N, dtype=self.DTYPE)
        t = np.zeros(N, dtype=self.DTYPE)
        oldMean = np.zeros(N, dtype=self.DTYPE)

        meanEnv = np.zeros(N, dtype=self.DTYPE)

        while(notFinish):
            self.logger.debug('IMF -- '+str(imfNo))

            yRes = (-(imf+corRes)).astype(self.DTYPE)
            tRes = (Res + yRes).astype(self.DTYPE)
            cor = (tRes - Res) - yRes
            Res = tRes

            #~ Res -= imf
            imf = Res.copy()
            if imf.dtype!=self.DTYPE: self.logger.error('imf.dtype: '+str(imf.dtype))

            mean = np.zeros(len(S), dtype=self.DTYPE)
            sumEnv = np.zeros(len(S), dtype=self.DTYPE)
            corEnv = np.zeros(len(S), dtype=self.DTYPE)

            # Counters
            n = 0   # All iterations for current imf.
            n_h = 0 # counts when |#zero - #ext| <=1

            # Start on-screen displaying
            if self.PLOT and self.INTERACTIVE:
                plt.ion()

            while(n<self.MAX_ITERATION):
                n += 1
                self.logger.debug("Iteration: "+str(n))

                maxPos, maxVal, minPos, minVal, indzer = self.find_extrema(T, imf)
                extNo = len(minPos)+len(maxPos)
                nzm = len(indzer)

                if extNo > 2:
                    # If scale tiny it might be noise, thus no need for
                    # further decomposition
                    if np.max(imf) - np.min(imf) < 1e-4:
                        msg = "Breaking decomposition as value range too small"
                        msg+= "\ndS: "+str(np.max(imf) - np.min(imf))
                        self.logger.info(msg)
                        notFinish = False
                        break

                    # Plotting. Either into file, or on-screen display.
                    if n>1 and self.PLOT:
                        plt.clf()
                        plt.plot(T, imf*scale, 'g')
                        plt.plot(T, maxEnv*scale, 'b')
                        plt.plot(T, minEnv*scale, 'r')
                        plt.plot(T, mean*scale, 'k--')
                        plt.title("imf{}_{:02}".format(imfNo, n-1))

                        if self.INTERACTIVE:
                            plt.draw()
                        else:
                            fName = "imf{}_{:02}".format(imfNo, n-1)
                            plt.savefig(os.path.join(self.plotPath,fName))

                    if Res.dtype!=self.DTYPE: self.logger.error('Res.dtype: '+str(Res.dtype))
                    if mean.dtype!=self.DTYPE: self.logger.error('mean.dtype: '+str(mean.dtype))

                    imfOld = imf.copy()
                    imf = Res - self.reduceScale*mean

                    maxEnv, minEnv, eMax, eMin = self.extract_max_min_spline(T, imf)

                    if type(maxEnv) == type(-1):
                        notFinish = True
                        break

                    y = 0.5*maxEnv - cor
                    t = sumEnv + y
                    cor = (t - sumEnv) - y
                    sumEnv = t

                    y = 0.5*minEnv - cor
                    t = sumEnv + y
                    cor = (t - sumEnv) - y
                    sumEnv = t

                    oldMean = mean.copy()
                    mean = sumEnv

                    # Fix number of iterations
                    if self.FIXE:
                        if n>=self.FIXE+1: break

                    # Fix number of iterations after number of zero-crossings
                    # and extrema differ at most by one.
                    elif self.FIXE_H:

                        maxPos, maxVal, minPos, minVal, indZer = self.find_extrema(T, imf)
                        extNo = len(maxPos)+len(minPos)
                        nzm = len(indZer)


                        if n == 1: continue
                        if abs(extNo-nzm)>1: n_h = 0
                        else:                n_h += 1

                        #if np.all(maxVal>0) and np.all(minVal<0):
                        #    n_h += 1
                        #else:
                        #    n_h = 0

                        # STOP
                        if n_h >= self.FIXE_H: break

                    # Stops after default stopping criteria are meet.
                    else:

                        maxPos, maxVal, minPos, minVal, indZer = self.find_extrema(T, imf)
                        extNo = len(maxPos) + len(minPos)
                        nzm = len(indZer)

                        f1 = self.check_imf(imf, maxEnv, minEnv, tmpMean, extNo)
                        #f2 = np.all(maxVal>0) and np.all(minVal<0)
                        f2 = abs(extNo - nzm)<2

                        # STOP
                        if f1 and f2: break

                else:
                    notFinish = False
                    break

            IMF[imfNo] = imf.copy() - cor
            imfNo += 1

            if self.end_condition(scaledS, IMF) or imfNo==maxImf:
                notFinish = False
                break

        #~ # Saving residuum
        #~ Res -= imf
        #~ #Res = scaledS - np.sum([IMF[i] for i in range(imfNo)],axis=0)
        #~ IMF[imfNo] = Res
        #~ imfNo += 1

        for key in list(IMF.keys()):
            IMF[key] *= scale
        nIMF = np.array([IMF[k] for k in sorted(IMF.keys())])
        return nIMF

###################################################
## Beginning of program

if __name__ == "__main__":

    import pylab as plt

    # Logging options
    logging.basicConfig(level=logging.DEBUG)

    # EMD options
    maxImf = -1
    DTYPE = np.float64

    # Signal options
    N = 400
    tMin, tMax = 0, 2*np.pi
    T = np.linspace(tMin, tMax, N, dtype=DTYPE)

    S = np.sin(20*T*(1+0.2*T)) + T**2 + np.sin(13*T)
    S = S.astype(DTYPE)
    print("Input S.dtype: "+str(S.dtype))

    # Prepare and run EMD
    emd = EMD()
    emd.PLOT = 0
    emd.FIXE_H = 5
    emd.nbsym = 2
    emd.splineKind = 'cubic'
    emd.DTYPE = DTYPE

    nIMF = emd.emd(S, T, maxImf)
    imfNo = nIMF.shape[0]

    # Plot results
    c = 1
    r = np.ceil((imfNo+1)/c)

    plt.ioff()
    plt.subplot(r,c,1)
    plt.plot(T, S, 'r')
    plt.xlim((tMin, tMax))
    plt.title("Original signal")

    for num in range(imfNo):
        plt.subplot(r,c,num+2)
        plt.plot(T, nIMF[num],'g')
        plt.xlim((tMin, tMax))
        plt.ylabel("Imf "+str(num+1))

    plt.tight_layout()
    plt.show()
