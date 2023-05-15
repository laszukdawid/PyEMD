#!/usr/bin/env python3
# Coding: UTF-8


import numpy as np
import matplotlib.pyplot as plt

from PyEMD import EMD
from PyEMD.splines import *
from PyEMD.utils import get_timeline


def test_spline(X,T,s_kind):
    """
    test the fitting with the ginven spline
    require:
     X: the signal, 1D numpy array
     T: time array. 1D numpy array of the same length of X
     s_kind: spline kind, string. can be one of the following splines:
     'akima', 'cubic', 'pchip'
    return:
     max_env: max spline envelope, 1D numpy array
     min_env: min spline envelope, 1D numpy array
     eMax: max extrema of the spline envelope
     eMin: min extrema of the spline envelope
    """

    emd = EMD()
    emd.spline_kind = s_kind
    max_env, min_env, eMax, eMin = emd.extract_max_min_spline(T,X)
    return max_env, min_env, eMax, eMin


def test_akima(X,T,ax):
    """
    test the fitting with akima spline
     require:
     X: the signal, 1D numpy array
     T: time array. 1D numpy array of the same length of X
     ax: matplotlib axis
    return:
     the plot of the spline envelope
     eMax: max extrema of the spline envelope
     eMin: min extrema of the spline envelope
    """

    max_env, min_env, eMax, eMin = test_spline(X, T, "akima")

    ax.plot(max_env,label='max akima')
    ax.plot(min_env,label='min akima')
    return eMax, eMin


def test_cubic(X,T,ax):
    """
    test the fitting with cubic spline
     require:
     X: the signal, 1D numpy array
     T: time array. 1D numpy array of the same length of X
     ax: matplotlib axis
    return:
     the plot of the spline envelope
     eMax: max extrema of the spline envelope
     eMin: min extrema of the spline envelope
    """

    max_env, min_env, eMax, eMin = test_spline(X, T, "cubic")

    ax.plot(max_env,label='max cubic')
    ax.plot(min_env,label='min cubic')
    return eMax, eMin


def test_pchip(X,T,ax):
    """
    test the fitting with pchip spline 
    'Piecewise Cubic Hermite Interpolating Polynomial'
     require:
     X: the signal, 1D numpy array
     T: time array. 1D numpy array of the same length of X
     ax: matplotlib axis
    return:
     the plot of the spline envelope
     eMax: max extrema of the spline envelope
     eMin: min extrema of the spline envelope
    """

    max_env, min_env, eMax, eMin = test_spline(X, T, "pchip")

    ax.plot(max_env,label='max pchip')
    ax.plot(min_env,label='min pchip')
    return eMax, eMin


def test_cubic_hermite(X,T,ax):
    """
    test the fitting with cubic_hermite spline 
     require:
     X: the signal, 1D numpy array
     T: time array. 1D numpy array of the same length of X
     ax: matplotlib axis
    return:
     the plot of the spline envelope
     eMax: max extrema of the spline envelope
     eMin: min extrema of the spline envelope
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