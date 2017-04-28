#!/usr/bin/python
# coding: UTF-8
#
# Author:   Dawid Laszuk
# Contact:  laszukdawid@gmail.com
#
# Feel free to contact for any information.

from __future__ import print_function

import logging
import time
import os

import numpy as np
import pylab as py
from scipy.interpolate import interp1d

class EMD:

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
        self.maxIteration = 400
        self.scaleFactor = 100

        self.PLOT = 0
        self.INTERACTIVE = 0
        self.plotPath = 'splineTest'

        self.splineKind = 'akima'

        self.DTYPE = np.float64
        self.FIXE = 0
        self.FIXE_H = 0

        self.MAX_ITERATION = 10000

    def splineHermite(self, T, P0, M0, P1, M1):
        # Normalized time - range (0,1)
        t = (T-T[0])/(T[-1]-T[0])
        t2 = t*t
        t3 = t2*t
        alfa = 0.2

        return P0*( 2*t3 - 3*t2 + 1) + \
               M0*(   t3 - 2*t2 + t)*alfa + \
               P1*(-2*t3 + 3*t2) + \
               M1*(   t3 -   t2)*alfa

    def spline_interplolate(self, T, P0, M0, P1, M1):
        """
        Spline interpolation: http://en.wikipedia.org/wiki/Spline_interpolation.
        """
        a = 1
        M0, M1 = a*M0, a*M1

        t0, t1 = T[0], T[-1]
        a =  M0*(t1-t0) - (P1-P0)
        b = -M1*(t1-t0) + (P1-P0)

        t = (T-T[0])/(T[-1]-T[0])

        # q = (1-t)*y0 + t*y1 + t*(1-t)*(a*(1-t) + b*t)
        return (1-t)*P0 + t*P1 + t*(1-t)*(a*(1-t) + b*t)

    def extractMaxMinSpline(self, T, S):
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

        # Create timeline
        tMin, tMax = T[0], T[-1]

        # Get indexes of extrema
        #~ maxPos, maxVal, minPos, minVal = self.findExtrema(T, S)
        #~ maxPos, maxVal, minPos, minVal, indzer = self.findExtrema_new(T, S)
        #~ maxPos, maxVal, minPos, minVal, indzer = self.findExtrema_simple(T, S)
        maxPos, maxVal, minPos, minVal, indzer = self.findExtremaGeneric(T, S)

        if maxPos.dtype!=self.DTYPE: self.logger.error('maxPos.dtype: '+str(maxPos.dtype))
        if maxVal.dtype!=self.DTYPE: self.logger.error('maxVal.dtype: '+str(maxVal.dtype))
        if minPos.dtype!=self.DTYPE: self.logger.error('minPos.dtype: '+str(minPos.dtype))
        if minVal.dtype!=self.DTYPE: self.logger.error('minVal.dtype: '+str(minVal.dtype))
        if len(maxPos) + len(minPos) < 3: return [-1]*4

        #########################################
        # Prepare spline
        order = 0
        dt = T[1]-T[0]
        #########################################

        # Extrapolation of signal (ober boundaries)
        maxExtrema, minExtrema = self.preparePoints(T, S, maxPos, maxVal, minPos, minVal)

        maxTSpline, maxSpline = self.splinePoints(T, maxExtrema, self.splineKind)
        minTSpline, minSpline = self.splinePoints(T, minExtrema, self.splineKind)

        if maxExtrema.dtype!=self.DTYPE: self.logger.error('maxExtrema.dtype: '+str(maxExtrema.dtype))
        if maxSpline.dtype!=self.DTYPE: self.logger.error('maxSpline.dtype: '+str(maxSpline.dtype))
        if maxTSpline.dtype!=self.DTYPE: self.logger.error('maxTSline.dtype: '+str(maxTSpline.dtype))

        return maxSpline, minSpline, maxExtrema, minExtrema

    def findExtremaGeneric(self, T, S):
        #~ return self.findExtrema(T,S)
        return self.findExtrema_simple(T,S)

    def preparePoints(self, T, S, maxPos, maxVal, minPos, minVal):
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
        leftExtType = {True:"max", False:"min"}[dPos<0]

        if (leftExtType == "max"):
            if (S[0]>minVal[0]) and len(maxPos)>1 and (np.abs(dPos)>(maxPos[0]-T[0])):
                # mirror signal to first extrem
                expandLeftMaxPos = 2*maxPos[0] - maxPos[1:nbsym+1]
                expandLeftMinPos = 2*maxPos[0] - minPos[0:nbsym]
                expandLeftMaxVal = maxVal[1:nbsym+1]
                expandLeftMinVal = minVal[0:nbsym]

            elif (S[0]>minVal[0]) and len(maxPos)>1 and (np.abs(dPos)<=(maxPos[0]-T[0])):
                # mirror signal to beginning
                expandLeftMaxPos = 2*T[0] - maxPos[0:nbsym]
                expandLeftMinPos = 2*T[0] - minPos[0:nbsym]
                expandLeftMaxVal = maxVal[0:nbsym]
                expandLeftMinVal = minVal[0:nbsym]

            else:
                # mirror signal to beginning
                expandLeftMaxPos = 2*T[0] - maxPos[0:nbsym]
                expandLeftMinPos = 2*T[0] - np.append(T[0], minPos[0:max(nbsym-1,0)])
                expandLeftMaxVal = maxVal[0:nbsym]
                expandLeftMinVal = np.append(S[0], minVal[0:max(nbsym-1,0)])


        elif (leftExtType == "min"):
            if (S[0] < maxVal[0]) and len(minPos)>1 and (np.abs(dPos)>(minPos[0]-T[0])):
                # mirror signal to first extrem
                expandLeftMaxPos = 2*minPos[0] - maxPos[0:nbsym]
                expandLeftMinPos = 2*minPos[0] - minPos[1:nbsym+1]
                expandLeftMaxVal = maxVal[0:nbsym]
                expandLeftMinVal = minVal[1:nbsym+1]

            elif (S[0] < maxVal[0]) and len(minPos)>1 and (np.abs(dPos)<(minPos[0]-T[0])):
                expandLeftMaxPos = 2*T[0] - maxPos[0:nbsym]
                expandLeftMinPos = 2*T[0] - minPos[0:nbsym]
                expandLeftMaxVal = maxVal[0:nbsym]
                expandLeftMinVal = minVal[0:nbsym]

            else:
                # mirror signal to begining
                expandLeftMaxPos = 2*T[0] - np.append(T[0], maxPos[0:max(nbsym-1,0)])
                expandLeftMinPos = 2*T[0] - minPos[0:nbsym]
                expandLeftMaxVal = np.append(S[0], maxVal[0:max(nbsym-1,0)])
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
        rightExtType = {True:"max", False:"min"}[dPos>0]

        if (rightExtType == "min"):
            if (S[-1] < maxVal[-1]) and len(minPos)>1 and (np.abs(dPos)>(T[-1]-minPos[-1])):
                # mirror signal to last extrem
                idxMax = max(0, endMax-nbsym)
                idxMin = max(0, endMin-nbsym-1)
                expandRightMaxPos = 2*minPos[-1] - maxPos[idxMax:]
                expandRightMinPos = 2*minPos[-1] - minPos[idxMin:-1]
                expandRightMaxVal = maxVal[idxMax:]
                expandRightMinVal = minVal[idxMin:-1]
            elif ((S[-1] < maxVal[-1]) and len(minPos)>1):
                # mirror signal to last extrem
                idxMax = max(0, endMax-nbsym)
                idxMin = max(0, endMin-nbsym)
                expandRightMaxPos = 2*T[-1] - maxPos[idxMax:]
                expandRightMinPos = 2*T[-1] - minPos[idxMin:]
                expandRightMaxVal = maxVal[idxMax:]
                expandRightMinVal = minVal[idxMin:]

            else:
                # mirror signal to the end
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

            elif (S[-1] > minVal[-1]) and len(maxPos>0):
                idxMax = max(0, endMax-nbsym)
                idxMin = max(0, endMin-nbsym)
                expandRightMaxPos = 2*T[-1] - maxPos[idxMax:]
                expandRightMinPos = 2*T[-1] - minPos[idxMin:]
                expandRightMaxVal = maxVal[idxMax:]
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

    def splinePoints(self, T, extrema, splineKind):
        """
        Constructs spline over given points.

        Input:
        ---------
            T: Time array.
            extrema: Poistion (1st row) and values (2nd row) of points.
            splineKind: Type of spline.

        Output:
        ---------
            T: Poistion array.
            spline: Spline over the given points.
        """

        kind = splineKind.lower()
        dt = T[1]-T[0]
        #~ t = T[np.r_[T>extrema[0,0]-dt] & np.r_[T<extrema[0,-1]+dt]]
        t = T[np.r_[T>=extrema[0,0]] & np.r_[T<=extrema[0,-1]]]

        if t.dtype != self.DTYPE: self.logger.error('t.dtype: '+str(t.dtype))
        if extrema.dtype != self.DTYPE: self.logger.error('extrema.dtype: '+str(extrema.dtype))

        if kind == "akima":
            return t, self.akima(extrema[0], extrema[1], t)

        elif kind == 'cubic':
            if extrema.shape[1]>3:
                return t, interp1d(extrema[0], extrema[1], kind=kind)(t).astype(self.DTYPE)
            else:
                return self.cubicSpline_3points(T, extrema)

        elif kind in ['slinear', 'quadratic', 'linear']:
            return T, interp1d(extrema[0], extrema[1], kind=kind)(t).astype(self.DTYPE)

        else:
            raise Exception("No such interpolation method!")

    def cubicSpline_3points(self, T, extrema):
        """
        Apperently scipy.interpolate.interp1d does not support
        cubic spline for less than 4 points.
        """

        x0, x1, x2 = extrema[0]
        y0, y1, y2 = extrema[1]

        x1x0, x2x1 = x1-x0, x2-x1
        y1y0, y2y1 = y1-y0, y2-y1
        _x1x0, _x2x1 = 1./x1x0, 1./x2x1

        m11, m12, m13= 2*_x1x0, _x1x0, 0
        m21, m22, m23 = _x1x0, 2.*(_x1x0+_x2x1), _x2x1
        m31, m32, m33 = 0, _x2x1, 2.*_x2x1

        v1 = 3*y1y0*_x1x0*_x1x0
        v3 = 3*y2y1*_x2x1*_x2x1
        v2 = v1+v3

        M = np.matrix([[m11,m12,m13],[m21,m22,m23],[m31,m32,m33]])
        v = np.matrix([v1,v2,v3]).T
        k = np.array(np.linalg.inv(M)*v)

        a1 = k[0]*x1x0 - y1y0
        b1 =-k[1]*x1x0 + y1y0
        a2 = k[1]*x2x1 - y2y1
        b2 =-k[2]*x2x1 + y2y1

        t = T[np.r_[T>=x0] & np.r_[T<=x2]]
        t1 = (T[np.r_[T>=x0]&np.r_[T< x1]] - x0)/x1x0
        t2 = (T[np.r_[T>=x1]&np.r_[T<=x2]] - x1)/x2x1
        t11, t22 = 1.-t1, 1.-t2

        q1 = t11*y0 + t1*y1 + t1*t11*(a1*t11 + b1*t1)
        q2 = t22*y1 + t2*y2 + t2*t22*(a2*t22 + b2*t2)
        q = np.append(q1,q2)

        return t, q.astype(self.DTYPE)

    def akima(self, X, Y, x):
        """
        Interpolates curve based on Akima's method [1].

        [1] H. Akima, "A new method of interpolation and smooth
            curve fitting based on local procedures", 1970.

        Input:
        ---------
            X: Position.
            Y: Values.
            x: Positions for interpolated spline.

        Output:
        ---------
            y: Interpolated spline.
        """

        n = len(X)
        if (len(X) != len(Y)):
            raise Exception('input x and y arrays must be of same length')

        dx = np.diff(X)
        dy = np.diff(Y)

        if dx.dtype != self.DTYPE: self.logger.error('dx.dtype: '+str(dx.dtype))

        if np.any(dx <= 0):
            raise Exception('input x-array must be in strictly ascending order')

        if np.any(x<X[0]) or np.any(x>X[-1]):
            raise Exception('All interpolation points xi must lie between x(1) and x(n)')

        # d - approximation of derivative
        # p, n - previous, next
        d = dy/dx
        if d.dtype != self.DTYPE: self.logger.error('d.dtype: '+str(d.dtype))

        dpp = 2*d[0]-d[1]
        dp = 2*dpp - d[0]

        dn = 2*d[n-2]-d[n-3]
        dnn = 2*dn-d[n-2]

        d1 = np.concatenate(([dpp], [dp], d, [dn], [dnn]))
        d1 = d1.astype(self.DTYPE)

        w = np.abs(np.diff(d1), dtype=self.DTYPE)
        # w1 = w_{i-1} = |d_{i+1}-d_{i}|
        # w2 = w_{i} = |d_{i-1}-d_{i-2}|
        w1, w2 = w[2:n+2], w[:n]
        w12 = w1+w2

        idx = np.nonzero(w12 > 1e-9*np.max(w12))[0]
        a1 = d1[1:n + 1].copy()

        a1[idx] = (w1[idx]*d1[idx+1] + w2[idx]*d1[idx+2])/w12[idx]
        a2 = (3.0*d - 2.0*a1[0:n-1] - a1[1:n]) / dx
        a3 = (a1[0:n-1] + a1[1:n] - 2.0*d) / (dx*dx)

        if a1.dtype != self.DTYPE: 'a1.dtype: ', a1.dtype
        if a2.dtype != self.DTYPE: 'a2.dtype: ', a2.dtype
        if a3.dtype != self.DTYPE: 'a3.dtype: ', a3.dtype

        bins = np.digitize(x, X)
        bins = np.minimum(bins, n - 1) - 1
        bb = bins[0:len(x)]
        _x = x - X[bb]

        out = ((_x*a3[bb] + a2[bb])*_x + a1[bb])*_x + Y[bb]

        if _x.dtype != self.DTYPE: self.logger.error('_x.dtype: '+str(_x.dtype))
        if out.dtype != self.DTYPE: self.logger.error('out.dtype: '+str(out.dtype))

        return out

    def findExtrema_new(self, t, s):
        dt = t[1]-t[0]
        scale = 2*dt*dt

        idx = self.notDuplicate(s)
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

        #################################
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
                indz = np.round((debz+finz)/2)
            else:
                indz = iz

            indzer = np.sort(np.append(indzer, indz))

        return localMaxPos, localMaxVal, localMinPos, localMinVal, indzer

    def notDuplicate(self, s):
        idx = [0]
        for i in range(1,len(s)-1):
            if (s[i] == s[i+1] and s[i] == s[i-1]):
               pass

            else: idx.append(i)
        idx.append(len(s)-1)
        return idx

    def findExtrema(self, t, s):
        """
        Estimates position and value of extrema by parabolical
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

        d = np.append(s[1:] - s[:-1],1)
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

        #################################
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
                indz = np.round((debz+finz)/2)
            else:
                indz = iz

            indzer = np.sort(np.append(indzer, indz))

        return localMaxPos, localMaxVal, localMinPos, localMinVal, indzer

    def findExtrema_simple(self, t, s):
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
                indz = np.round((debz+finz)/2)
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
                            imax.append(round((fins[k]+debs[k])/2.))
                    else:
                        if d[fins[k]] > 0:
                            imin.append(round((fins[k]+debs[k])/2.))

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

    def endCondition(self, Res, IMF):
        # When to stop EMD
        tmp = Res.copy()
        for imfNo in list(IMF.keys()):
            tmp -= IMF[imfNo]

        #~ # Power is enought
        #~ if np.log10(np.abs(tmp).sum()/np.abs(Res).sum()) < powerThreshold:
            #~ self.logger.info("FINISHED -- POWER RATIO")
            #~ return True

        if np.max(tmp) - np.min(tmp) < self.rangeThreshold:
            self.logger.info("FINISHED -- RANGE")
            return True

        if np.sum(np.abs(tmp)) < self.totalPowerThreshold:
            self.logger.info("FINISHED -- SUM POWER")
            return True

    def checkImf(self, imfNew, imfOld, eMax, eMin, mean):
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
            #~ self.logger.info("Scaled variance -- PASSED")
            return True
        elif std < self.stdThreshold:
            #~ self.logger.info("Standard deviation -- PASSED")
            return True
        else:
            return False

    def _common_dtype(self, x, y):

        dtype = np.find_common_type([x.dtype, y.dtype], [])
        if x.dtype != dtype: x = x.astype(dtype)
        if y.dtype != dtype: y = y.astype(dtype)

        return x, y

    def emd(self, S, timeLine=None, maxImf=None):
        """
        Performs Emerical Mode Decomposition on signal S.
        The decomposition is limited to maxImf imf. No limitation as default.
        Returns IMF functions in dic format. IMF = {0:imf0, 1:imf1...}.

        Input:
        ---------
            S: Signal.
            timeLine: Positions of signal. If none passed numpy arange is created.
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

        if timeLine is None: timeLine = np.arange(len(S))
        if maxImf is None: maxImf = -1

        # Make sure same types are dealt
        S, timeLine = self._common_dtype(S, timeLine)
        self.DTYPE = S.dtype

        Res = S.astype(self.DTYPE)
        scale = (max(Res) - min(Res))/self.scaleFactor
        Res, scaledS = Res/scale, S/scale
        imf = np.zeros(len(S), dtype=self.DTYPE)
        imfOld = Res.copy()

        N = len(S)

        if Res.dtype!=self.DTYPE: self.logger.error('Res.dtype: '+str(Res.dtype))
        if scaledS.dtype!=self.DTYPE: self.logger.error('scaledS.dtype: '+str(scaledS.dtype))
        if imf.dtype!=self.DTYPE: self.logger.error('imf.dtype: '+str(imf.dtype))
        if imfOld.dtype!=self.DTYPE: self.logger.error('imfOld.dtype: '+str(imfOld.dtype))
        if timeLine.dtype!=self.DTYPE: self.logger.error('timeLine.dtype: '+str(timeLine.dtype))

        if S.shape != timeLine.shape:
            info = "Time array should be the same size as signal."
            raise Exception(info)

        # Create arrays
        IMF = {} # Dic for imfs signals
        EXT = {} # Dic for number of extrema
        TIME = {} # Dic for time computing of single imf
        ITER = {} # Dic for number of iterations
        imfNo = 0
        notFinish = True

        time0 = time.time()

        corRes = np.zeros(N, dtype=self.DTYPE)
        y = np.zeros(N, dtype=self.DTYPE)
        t = np.zeros(N, dtype=self.DTYPE)
        oldMean = np.zeros(N, dtype=self.DTYPE)

        meanEnv = np.zeros(N, dtype=self.DTYPE)

        while(notFinish):
            self.logger.debug('IMF -- '+str(imfNo))

            #~ yRes = (-imf - corRes).astype(self.DTYPE)
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
            if sumEnv.dtype != self.DTYPE: self.logger.error('sumEnv.dtype: '+str(sumEnv.dtype))
            if corEnv.dtype != self.DTYPE: self.logger.error('corEnv.dtype: '+str(corEnv.dtype))

            # Counters
            n = 0   # All iterations for current imf.
            n_h = 0 # counts when |#zero - #ext| <=1

            t0 = time.time()

            # Start on-screen displaying
            if self.PLOT and self.INTERACTIVE:
                py.ion()

            while(n<self.MAX_ITERATION):
                n += 1

                #~ maxPos, maxVal, minPos, minVal, indzer = self.findExtrema_simple(timeLine, imf)
                maxPos, maxVal, minPos, minVal, indzer = self.findExtremaGeneric(timeLine, imf)
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
                        py.clf()
                        py.plot(timeLine, imf*scale, 'g')
                        py.plot(timeLine, maxEnv*scale, 'b')
                        py.plot(timeLine, minEnv*scale, 'r')
                        py.plot(timeLine, mean*scale, 'k--')
                        if self.INTERACTIVE:
                            py.draw()
                        else:
                            fName = "imf{}_{:02}".format(imfNo, n-1)
                            py.savefig(os.path.join(self.plotPath,fName))

                    if Res.dtype!=self.DTYPE: self.logger.error('Res.dtype: '+str(Res.dtype))
                    if mean.dtype!=self.DTYPE: self.logger.error('mean.dtype: '+str(mean.dtype))

                    imfOld = imf.copy()
                    imf = Res - self.reduceScale*mean

                    maxEnv, minEnv, eMax, eMin = self.extractMaxMinSpline(timeLine, imf)
                    tmpMean = 0.5*(maxEnv+minEnv)

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

                        #~ maxPos, maxVal, minPos, minVal, indZer = self.findExtrema_simple(timeLine, imf)
                        maxPos, maxVal, minPos, minVal, indZer = self.findExtremaGeneric(timeLine, imf)
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

                        #~ maxPos, maxVal, minPos, minVal, indZer = self.findExtrema_simple(timeLine, imf)
                        maxPos, maxVal, minPos, minVal, indZer = self.findExtremaGeneric(timeLine, imf)
                        extNo = len(maxPos) + len(minPos)
                        nzm = len(indZer)

                        f1 = self.checkImf(imf, maxEnv, minEnv, tmpMean, extNo)
                        #f2 = np.all(maxVal>0) and np.all(minVal<0)
                        f2 = abs(extNo - nzm)<2

                        # STOP
                        if f1 and f2: break

                else:
                    EXT[imfNo] = extNo
                    notFinish = False
                    break

            #IMF[imfNo] = imf.copy() - 0.5*(corEnvMax+corEnvMin)
            IMF[imfNo] = imf.copy() - cor
            ITER[imfNo] = n
            EXT[imfNo] = extNo
            TIME[imfNo] = time.time() - t0
            imfNo += 1

            if self.endCondition(scaledS, IMF) or imfNo==maxImf:
                notFinish = False
                break

        #~ # Saving residuum
        #~ Res -= imf
        #~ #Res = scaledS - np.sum([IMF[i] for i in range(imfNo)],axis=0)
        #~ IMF[imfNo] = Res
        #~ ITER[imfNo] = 0
        #~ EXT[imfNo] = self.getExtremaNo(Res)
        #~ TIME[imfNo] = 0
        #~ imfNo += 1
        time1 = time.time()

        for key in list(IMF.keys()):
            IMF[key] *= scale
        #return IMF, EXT, TIME, ITER, imfNo
        return IMF, EXT, ITER, imfNo

###################################################
## Beggining of program

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    N = 1000
    maxImf = -1

    TYPE = 32
    if   TYPE == 16: DTYPE = np.float16
    elif TYPE == 32: DTYPE = np.float32
    elif TYPE == 64: DTYPE = np.float64
    else:            DTYPE = np.float64

    t = np.linspace(1*np.pi, 4*np.pi, N)
    timeLine = t.astype(DTYPE)
    t = t.astype(DTYPE)

    sin = np.sin
    cos = np.cos
    #tS = 'sin(3*t**(-1)) + cos(2*t)'
    #tS = 'sin(3*t**1.4 + cos(4*t))*cos(8*t + sin(6*t))/t'
    #tS = 't**(sin(2.76*t)-cos(t**2))'
    #~ tS = 'sin(3*t**1.2)**2 + cos(2.2*t**1.7)'
    tS = 'sin(3*t**1.2)**2 + cos(2.2*t**(t/8))'
    S = eval(tS)
    S = S.astype(DTYPE)
    print("Input S.dtype: "+str(S.dtype))

    #~ S = np.random.random(len(t))
    #~ S = mvAvg(S[::-1], 5)
    #~ S = mvAvg(S[::-1], 5)
    emd = EMD()
    emd.PLOT = 0
    emd.FIXE_H = 5
    emd.nbsym = 2
    emd.splineKind = 'cubic'
    emd.DTYPE = DTYPE
    #IMF, EXT, TIME, ITER, imfNo = emd.emd(S, timeLine, maxImf)
    IMF, EXT, ITER, imfNo = emd.emd(S, timeLine, maxImf)

    c = np.floor(np.sqrt(imfNo+3))
    r = np.ceil( (imfNo+3)/c)

    py.ioff()
    py.subplot(r,c,1)
    py.plot(timeLine, S, 'r')
    py.title("Original signal")

    py.subplot(r,c,2)
    py.plot([EXT[i] for i in range(imfNo)], 'o')
    py.title("Number of extrema")

    py.subplot(r,c,3)
    py.plot([ITER[i] for i in range(imfNo)], 'o')
    py.title("Number of iterations")

    def extF(s):
        state1 = np.r_[np.abs(s[1:-1]) > np.abs(s[:-2])]
        state2 = np.r_[np.abs(s[1:-1]) > np.abs(s[2:])]
        return np.arange(1,len(s)-1)[state1 & state2]

    for num in range(imfNo):
        py.subplot(r,c,num+4)
        py.plot(timeLine, IMF[num],'g')
        #~ py.plot(timeLine[extF(IMF[num])], IMF[num][extF(IMF[num])],'ok')
        py.title("Imf no " +str(num) )

    py.tight_layout()
    py.show()
