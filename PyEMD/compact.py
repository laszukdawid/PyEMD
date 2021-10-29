import numpy as np


def TDMAsolver(a, b, c, d):
    """Thomas algorithm to solve tridiagonal linear systems with
    non-periodic BC.

    | b0  c0                 | | . |     | . |
    | a1  b1  c1             | | . |     | . |
    |     a2  b2  c2         | | x |  =  | d |
    |         ..........     | | . |     | . |
    |             an  bn  cn | | . |     | . |
    """
    n = len(b)

    cp = np.zeros(n)
    cp[0] = c[0] / b[0]
    for i in range(1, n - 1):
        cp[i] = c[i] / (b[i] - a[i] * cp[i - 1])

    dp = np.zeros(n)
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        dp[i] = (d[i] - a[i] * dp[i - 1]) / (b[i] - a[i] * cp[i - 1])

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def filt6(f, alpha):
    """
    6th Order compact filter (non-periodic BC).

    References:
    -----------
    Lele, S. K. - Compact finite difference schemes with spectral-like
    resolution. Journal of Computational Physics 103 (1992) 16-42

    Visbal, M. R. and Gaitonde, D. V. - On the use of higher-order finite-
    difference schemes on curvilinear and deforming meshes. Journal of
    Computational Physics 181 (2002) 155-185
    """
    Ca = (11.0 + 10.0 * alpha) / 16.0
    Cb = (15.0 + 34.0 * alpha) / 32.0
    Cc = (-3.0 + 6.0 * alpha) / 16.0
    Cd = (1.0 - 2.0 * alpha) / 32.0

    n = len(f)

    rhs = np.zeros(n)

    rhs[3:-3] = (
        Cd * 0.5 * (f[6:] + f[:-6]) + Cc * 0.5 * (f[5:-1] + f[1:-5]) + Cb * 0.5 * (f[4:-2] + f[2:-4]) + Ca * f[3:-3]
    )

    # Non-periodic BC:
    rhs[0] = (15.0 / 16.0) * f[0] + (4.0 * f[1] - 6.0 * f[2] + 4.0 * f[3] - f[4]) / 16.0

    rhs[1] = (3.0 / 4.0) * f[1] + (f[0] + 6.0 * f[2] - 4.0 * f[3] + f[4]) / 16.0

    rhs[2] = (5.0 / 8.0) * f[2] + (-f[0] + 4.0 * f[1] + 4.0 * f[3] - f[4]) / 16.0

    rhs[-1] = (15.0 / 16.0) * f[-1] + (4.0 * f[-2] - 6.0 * f[-3] + 4.0 * f[-4] - f[-5]) / 16.0

    rhs[-2] = (3.0 / 4.0) * f[-2] + (f[-1] + 6.0 * f[-3] - 4.0 * f[-4] + f[-5]) / 16.0

    rhs[-3] = (5.0 / 8.0) * f[-3] + (-f[-1] + 4.0 * f[-2] + 4.0 * f[-4] - f[-5]) / 16.0

    Da = alpha * np.ones(n)
    Db = np.ones(n)
    Dc = alpha * np.ones(n)

    # 1st point
    Dc[0] = 0.0
    # 2nd point
    Da[1] = Dc[1] = 0.0
    # 3rd point
    Da[2] = Dc[2] = 0.0
    # last point
    Da[-1] = 0.0
    # 2nd from last
    Da[-2] = Dc[-2] = 0.0
    # 3rd from last
    Da[-3] = Dc[-3] = 0.0

    return TDMAsolver(Da, Db, Dc, rhs)


def pade6(vec, h):
    """
    6th Order compact finite difference scheme (non-periodic BC).

    Lele, S. K. - Compact finite difference schemes with spectral-like
    resolution. Journal of Computational Physics 103 (1992) 16-42
    """
    n = len(vec)
    rhs = np.zeros(n)

    a = 14.0 / 18.0
    b = 1.0 / 36.0

    rhs[2:-2] = (vec[3:-1] - vec[1:-3]) * (a / h) + (vec[4:] - vec[0:-4]) * (b / h)

    # boundaries:
    rhs[0] = (
        (-197.0 / 60.0) * vec[0]
        + (-5.0 / 12.0) * vec[1]
        + 5.0 * vec[2]
        + (-5.0 / 3.0) * vec[3]
        + (5.0 / 12.0) * vec[4]
        + (-1.0 / 20.0) * vec[5]
    ) / h

    rhs[1] = (
        (-20.0 / 33.0) * vec[0]
        + (-35.0 / 132.0) * vec[1]
        + (34.0 / 33.0) * vec[2]
        + (-7.0 / 33.0) * vec[3]
        + (2.0 / 33.0) * vec[4]
        + (-1.0 / 132.0) * vec[5]
    ) / h

    rhs[-1] = (
        (197.0 / 60.0) * vec[-1]
        + (5.0 / 12.0) * vec[-2]
        + (-5.0) * vec[-3]
        + (5.0 / 3.0) * vec[-4]
        + (-5.0 / 12.0) * vec[-5]
        + (1.0 / 20.0) * vec[-6]
    ) / h

    rhs[-2] = (
        (20.0 / 33.0) * vec[-1]
        + (35.0 / 132.0) * vec[-2]
        + (-34.0 / 33.0) * vec[-3]
        + (7.0 / 33.0) * vec[-4]
        + (-2.0 / 33.0) * vec[-5]
        + (1.0 / 132.0) * vec[-6]
    ) / h

    alpha1 = 5.0  # j = 1 and n
    alpha2 = 2.0 / 11  # j = 2 and n-1
    alpha = 1.0 / 3.0

    Db = np.ones(n)
    Da = alpha * np.ones(n)
    Dc = alpha * np.ones(n)

    # boundaries:
    Da[1] = alpha2
    Da[-1] = alpha1
    Da[-2] = alpha2
    Dc[0] = alpha1
    Dc[1] = alpha2
    Dc[-2] = alpha2

    return TDMAsolver(Da, Db, Dc, rhs)
