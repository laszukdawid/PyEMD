import numpy as np
from scipy.interpolate import Akima1DInterpolator


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
    # k = np.array(tmp)

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

    return t, q


def akima(X, Y, x):
    spl = Akima1DInterpolator(X, Y)
    return spl(x)


# class Akima1DInterpolator(CubicHermiteSpline):
#     """
#     Akima interpolator
#     Fit piecewise cubic polynomials, given vectors x and y. The interpolation
#     method by Akima uses a continuously differentiable sub-spline built from
#     piecewise cubic polynomials. The resultant curve passes through the given
#     data points and will appear smooth and natural.
#     Parameters
#     ----------
#     x : ndarray, shape (m, )
#         1-D array of monotonically increasing real values.
#     y : ndarray, shape (m, ...)
#         N-D array of real values. The length of ``y`` along the first axis
#         must be equal to the length of ``x``.
#     axis : int, optional
#         Specifies the axis of ``y`` along which to interpolate. Interpolation
#         defaults to the first axis of ``y``.
#     Methods
#     -------
#     __call__
#     derivative
#     antiderivative
#     roots
#     See Also
#     --------
#     PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
#     CubicSpline : Cubic spline data interpolator.
#     PPoly : Piecewise polynomial in terms of coefficients and breakpoints
#     Notes
#     -----
#     .. versionadded:: 0.14
#     Use only for precise data, as the fitted curve passes through the given
#     points exactly. This routine is useful for plotting a pleasingly smooth
#     curve through a few given points for purposes of plotting.
#     References
#     ----------
#     [1] A new method of interpolation and smooth curve fitting based
#         on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4),
#         589-602.
#     """

#     def __init__(self, x, y, axis=0):
#         # Original implementation in MATLAB by N. Shamsundar (BSD licensed), see
#         # https://www.mathworks.com/matlabcentral/fileexchange/1814-akima-interpolation
#         x, dx, y, axis, _ = prepare_input(x, y, axis)
#         # determine slopes between breakpoints
#         m = np.empty((x.size + 3, ) + y.shape[1:])
#         dx = dx[(slice(None), ) + (None, ) * (y.ndim - 1)]
#         m[2:-2] = np.diff(y, axis=0) / dx

#         # add two additional points on the left ...
#         m[1] = 2. * m[2] - m[3]
#         m[0] = 2. * m[1] - m[2]
#         # ... and on the right
#         m[-2] = 2. * m[-3] - m[-4]
#         m[-1] = 2. * m[-2] - m[-3]

#         # if m1 == m2 != m3 == m4, the slope at the breakpoint is not defined.
#         # This is the fill value:
#         t = .5 * (m[3:] + m[:-3])
#         # get the denominator of the slope t
#         dm = np.abs(np.diff(m, axis=0))
#         f1 = dm[2:]
#         f2 = dm[:-2]
#         f12 = f1 + f2
#         # These are the mask of where the slope at breakpoint is defined:
#         ind = np.nonzero(f12 > 1e-9 * np.max(f12))
#         x_ind, y_ind = ind[0], ind[1:]
#         # Set the slope at breakpoint
#         t[ind] = (f1[ind] * m[(x_ind + 1,) + y_ind] +
#                   f2[ind] * m[(x_ind + 2,) + y_ind]) / f12[ind]

#         super().__init__(x, y, t, axis=0, extrapolate=False)
#         self.axis = axis

#     def extend(self, c, x, right=True):
#         raise NotImplementedError("Extending a 1-D Akima interpolator is not "
#                                   "yet implemented")

#     # These are inherited from PPoly, but they do not produce an Akima
#     # interpolator. Hence stub them out.
#     @classmethod
#     def from_spline(cls, tck, extrapolate=None):
#         raise NotImplementedError("This method does not make sense for "
#                                   "an Akima interpolator.")

#     @classmethod
#     def from_bernstein_basis(cls, bp, extrapolate=None):
#         raise NotImplementedError("This method does not make sense for "
#                                   "an Akima interpolator.")

# class CubicHermiteSpline(PPoly):
#     """Piecewise-cubic interpolator matching values and first derivatives.
#     The result is represented as a `PPoly` instance.
#     Parameters
#     ----------
#     x : array_like, shape (n,)
#         1-D array containing values of the independent variable.
#         Values must be real, finite and in strictly increasing order.
#     y : array_like
#         Array containing values of the dependent variable. It can have
#         arbitrary number of dimensions, but the length along ``axis``
#         (see below) must match the length of ``x``. Values must be finite.
#     dydx : array_like
#         Array containing derivatives of the dependent variable. It can have
#         arbitrary number of dimensions, but the length along ``axis``
#         (see below) must match the length of ``x``. Values must be finite.
#     axis : int, optional
#         Axis along which `y` is assumed to be varying. Meaning that for
#         ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
#         Default is 0.
#     extrapolate : {bool, 'periodic', None}, optional
#         If bool, determines whether to extrapolate to out-of-bounds points
#         based on first and last intervals, or to return NaNs. If 'periodic',
#         periodic extrapolation is used. If None (default), it is set to True.
#     Attributes
#     ----------
#     x : ndarray, shape (n,)
#         Breakpoints. The same ``x`` which was passed to the constructor.
#     c : ndarray, shape (4, n-1, ...)
#         Coefficients of the polynomials on each segment. The trailing
#         dimensions match the dimensions of `y`, excluding ``axis``.
#         For example, if `y` is 1-D, then ``c[k, i]`` is a coefficient for
#         ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.
#     axis : int
#         Interpolation axis. The same axis which was passed to the
#         constructor.
#     Methods
#     -------
#     __call__
#     derivative
#     antiderivative
#     integrate
#     roots
#     See Also
#     --------
#     Akima1DInterpolator : Akima 1D interpolator.
#     PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
#     CubicSpline : Cubic spline data interpolator.
#     PPoly : Piecewise polynomial in terms of coefficients and breakpoints
#     Notes
#     -----
#     If you want to create a higher-order spline matching higher-order
#     derivatives, use `BPoly.from_derivatives`.
#     References
#     ----------
#     .. [1] `Cubic Hermite spline
#             <https://en.wikipedia.org/wiki/Cubic_Hermite_spline>`_
#             on Wikipedia.
#     """
#     def __init__(self, x, y, dydx, axis=0, extrapolate=None):
#         if extrapolate is None:
#             extrapolate = True

#         x, dx, y, axis, dydx = prepare_input(x, y, axis, dydx)

#         dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
#         slope = np.diff(y, axis=0) / dxr
#         t = (dydx[:-1] + dydx[1:] - 2 * slope) / dxr

#         c = np.empty((4, len(x) - 1) + y.shape[1:], dtype=t.dtype)
#         c[0] = t / dxr
#         c[1] = (slope - dydx[:-1]) / dxr - t
#         c[2] = dydx[:-1]
#         c[3] = y[:-1]

#         super().__init__(c, x, extrapolate=extrapolate)
#         self.axis = axis

# def prepare_input(x, y, axis, dydx=None):
#     """Prepare input for cubic spline interpolators.
#     All data are converted to numpy arrays and checked for correctness.
#     Axes equal to `axis` of arrays `y` and `dydx` are moved to be the 0th
#     axis. The value of `axis` is converted to lie in
#     [0, number of dimensions of `y`).
#     """

#     x, y = map(np.asarray, (x, y))
#     if np.issubdtype(x.dtype, np.complexfloating):
#         raise ValueError("`x` must contain real values.")
#     x = x.astype(float)

#     if np.issubdtype(y.dtype, np.complexfloating):
#         dtype = complex
#     else:
#         dtype = float

#     if dydx is not None:
#         dydx = np.asarray(dydx)
#         if y.shape != dydx.shape:
#             raise ValueError("The shapes of `y` and `dydx` must be identical.")
#         if np.issubdtype(dydx.dtype, np.complexfloating):
#             dtype = complex
#         dydx = dydx.astype(dtype, copy=False)

#     y = y.astype(dtype, copy=False)
#     axis = axis % y.ndim
#     if x.ndim != 1:
#         raise ValueError("`x` must be 1-dimensional.")
#     if x.shape[0] < 2:
#         raise ValueError("`x` must contain at least 2 elements.")
#     if x.shape[0] != y.shape[axis]:
#         raise ValueError("The length of `y` along `axis`={0} doesn't "
#                          "match the length of `x`".format(axis))

#     if not np.all(np.isfinite(x)):
#         raise ValueError("`x` must contain only finite values.")
#     if not np.all(np.isfinite(y)):
#         raise ValueError("`y` must contain only finite values.")

#     if dydx is not None and not np.all(np.isfinite(dydx)):
#         raise ValueError("`dydx` must contain only finite values.")

#     dx = np.diff(x)
#     if np.any(dx <= 0):
#         raise ValueError("`x` must be strictly increasing sequence.")

#     y = np.moveaxis(y, axis, 0)
#     if dydx is not None:
#         dydx = np.moveaxis(dydx, axis, 0)

#     return x, dx, y, axis,
