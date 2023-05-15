#!/usr/bin/env python3
# Coding: UTF-8


import numpy as np
import matplotlib.pyplot as plt

from PyEMD import EMD
from PyEMD.splines import *
from PyEMD.utils import get_timeline


def test_spline(X,T,s_kind):
    """
    Test the fitting with the given spline.

    Parameters
    ----------
    X : 1D numpy array
        the signal
    T : 1D numpy array
        Position or time array. It has the same length as X
    s_kind : string
        spline kind. can be one of the following splines:
        'akima', 'cubic', 'pchip', 'cubic_hermite'

    Returns
    -------
    max_env : 1D numpy array
        max spline envelope
    min_env : 1D numpy array
        min spline envelope
    eMax : numpy array
        max extrema of the signal
    eMin : numpy array
        min extrema of the signal
    """

    emd = EMD()
    emd.spline_kind = s_kind
    max_env, min_env, eMax, eMin = emd.extract_max_min_spline(T,X)
    return max_env, min_env, eMax, eMin


def test_akima(X,T,ax):
    """
    test the fitting with akima spline.

    Parameters
    ----------
    X : 1D numpy array
        the signal
    T : 1D numpy array
        Position or time array. It has the same length as X
    ax : matplotlib axis
        the axis used for plotting
    Returns
    -------
    eMax : numpy array
        max extrema of the signal
    eMin : numpy array
        min extrema of the signal
     the plot of the spline envelope
    """

    max_env, min_env, eMax, eMin = test_spline(X, T, "akima")

    ax.plot(max_env,label='max akima')
    ax.plot(min_env,label='min akima')
    return eMax, eMin


def test_cubic(X,T,ax):
    """
    test the fitting with cubic spline
    
    Parameters
    ----------
    see 'test_akima'

    Returns
    -------
    see 'test_akima'
    """

    max_env, min_env, eMax, eMin = test_spline(X, T, "cubic")

    ax.plot(max_env,label='max cubic')
    ax.plot(min_env,label='min cubic')
    return eMax, eMin


def test_pchip(X,T,ax):
    """
    test the fitting with pchip spline 
    'Piecewise Cubic Hermite Interpolating Polynomial'

    Parameters
    ----------
    see 'test_akima'

    Returns
    -------
    see 'test_akima'
    """

    max_env, min_env, eMax, eMin = test_spline(X, T, "pchip")

    ax.plot(max_env,label='max pchip')
    ax.plot(min_env,label='min pchip')
    return eMax, eMin


def test_cubic_hermite(X,T,ax):
    """
    test the fitting with cubic_hermite spline 

    Parameters
    ----------
    see 'test_akima'

    Returns
    -------
    see 'test_akima'
    """

    max_env, min_env, eMax, eMin = test_spline(X, T, "cubic_hermite")

    ax.plot(max_env,label='max cubic_hermite')
    ax.plot(min_env,label='min cubic_hermite')
    return eMax, eMin



if __name__ == "__main__":

    X = np.random.normal(size=200)
    T = get_timeline(len(X),X.dtype)
    T = EMD._normalize_time(T)

    fig, ax = plt.subplots()
    ax.plot(X,'--',lw=2,c='k')
    emax_akima, emin_akima = test_akima(X, T, ax)
    emax_cubic, emin_cubic = test_cubic(X, T, ax)
    emax_pchip, emin_pchip = test_pchip(X, T, ax)
    emax_chermite, emin_chermite = test_cubic_hermite(X, T, ax)
    ax.plot(emax_akima[0],emax_akima[1],'--')
    ax.legend()
    plt.show()