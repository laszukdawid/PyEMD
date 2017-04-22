from __future__ import division
import numpy as np

def cubic_spline_3pts(x, y, T):
    """
    Apperently scipy.interpolate.interp1d does not support
    cubic spline for less than 4 points.
    """

    x0, x1, x2 = x
    y0, y1, y2 = y

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

    return t, q

def akima(X, Y, x):
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
        raise Exception('Input x and y arrays must be of same length')

    X = np.array(X)
    Y = np.array(Y)

    dx = np.diff(X)
    dy = np.diff(Y)

    if np.any(dx <= 0):
        raise Exception('Input x-array must be in strictly ascending order')

    if x[0]<X[0] or x[-1]>X[-1]:
        msg = 'All interpolation points xi must lie between x[0] and x[-1]'
        raise Exception(msg)

    x = np.array(x)

    # d - approximation of derivative
    # p, n - previous, next
    d = dy/dx

    dpp = 2*d[0]-d[1]
    dp = 2*dpp - d[0]

    dn = 2*d[n-2]-d[n-3]
    dnn = 2*dn-d[n-2]

    d1 = np.concatenate(([dpp], [dp], d, [dn], [dnn]))

    w = np.abs(np.diff(d1))
    # w1 = w_{i-1} = |d_{i+1}-d_{i}|
    # w2 = w_{i} = |d_{i-1}-d_{i-2}|
    w1, w2 = w[2:n+2], w[:n]
    w12 = w1+w2

    idx = np.nonzero(w12 > 1e-9*np.max(w12))[0]
    a1 = d1[1:n + 1].copy()

    a1[idx] = (w1[idx]*d1[idx+1] + w2[idx]*d1[idx+2])/w12[idx]
    a2 = (3.0*d - 2.0*a1[0:n-1] - a1[1:n]) / dx
    a3 = (a1[0:n-1] + a1[1:n] - 2.0*d) / (dx*dx)

    bins = np.digitize(x, X)
    bins = np.minimum(bins, n - 1) - 1
    bb = bins[0:len(x)]
    _x = x - X[bb]

    out = ((_x*a3[bb] + a2[bb])*_x + a1[bb])*_x + Y[bb]

    return out

