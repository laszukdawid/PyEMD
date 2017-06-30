#!/usr/bin/python
# coding: UTF-8
#
# Author:   Dawid Laszuk
# Contact:  laszukdawid@gmail.com
#
# Edited:   20/06/2017
#
# Feel free to contact for any information.

from __future__ import division, print_function

import logging
import numpy as np
import os

#from scipy.ndimage import maximum_filter
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.interpolate import SmoothBivariateSpline as SBS

class EMD2D:
    """
    **Empirical Mode Decompoition** on 2D objects like images.

    """

    logger = logging.getLogger(__name__)

    def __init__(self, nbsym=2, **kwargs):
        # Declare constants
        self.std_thr = 0.2
        self.svar_thr = 0.001
        self.power_thr = -5
        self.total_power_thr = 0.01
        self.range_thr = 0.001

        self.nbsym = nbsym
        self.reduce_scale = 1.
        self.scale_factor = 1.

        self.PLOT = 0
        self.INTERACTIVE = 0
        self.plotPath = 'splineTest'

        self.DTYPE = np.float64
        self.FIXE = 0
        self.FIXE_H = 0

        self.MAX_ITERATION = 1000

        # Update based on options
        for key in kwargs.keys():
            if key in self.__dict__.keys():
                self.__dict__[key] = kwargs[key]

        if self.PLOT:
            import pylab as plt


    def extract_max_min_spline(self, T, S):
        return True

    def prepare_points(self, image):
        """Extrapolates how extrapolation should behave on edges."""
        return image

    def spline_points(self, X, Y, Z, xi, yi):
        """Interpolates for given set of points"""

        # SBS requires at least m=(kx+1)*(ky+1) points,
        # where kx=ky=3 (default) is the degree of bivariate spline.
        # Thus, if less than 16=(3+1)*(3+1) points, adjust kx & ky.
        k = {}
        if X.size < 16:
            k['kx'] = int(np.floor(np.sqrt(len(X)))-1)
            k['ky'] = k['kx']

        spline = SBS(X, Y, Z, **k)
        return spline(xi, yi)

    def find_extrema(self, image):
	"""
	takes an image and detect the peaks usingthe local maximum filter.
	returns a boolean mask of the peaks (i.e. 1 when
	the pixel's value is the neighborhood maximum, 0 otherwise)
	"""

	# define an 8-connected neighborhood
	neighborhood = generate_binary_structure(2,2)

	#apply the local maximum filter; all pixel of maximal value 
	#in their neighborhood are set to 1
	local_min = maximum_filter(-image, footprint=neighborhood)==-image
	local_max = maximum_filter(image, footprint=neighborhood)==image

	# can't distinguish between background zero and filter zero
	background = (image==0)

	#appear along the background border (artifact of the local maximum filter)
	eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

	#we obtain the final mask, containing only peaks, 
	#by removing the background from the local_max mask (xor operation)
	min_peaks = local_min ^ eroded_background
	max_peaks = local_max ^ eroded_background

        min_peaks[[0,-1],:] = False
        min_peaks[:,[0,-1]] = False
        max_peaks[[0,-1],:] = False
        max_peaks[:,[0,-1]] = False

	return min_peaks, max_peaks

    def end_condition(self, img, IMFs):
        rec = np.sum(IMFs, axis=0)

        # If reconstruction is perfect, no need for more tests
        if np.allclose(img, rec):
            return True

        return False

    def check_imf(self, imf_new, imf_old, eMax, eMin, mean):
        return True

    def emd(self, img, max_imf=-1):
        return img

########################################
if __name__ == "__main__":
    x = np.arange(50)
    y = np.arange(50).reshape((-1,1))

    pi2 = 2*np.pi
    img = np.sin(x*pi2)*np.cos(y*3*pi2)+x
    e = extrema2D(img)
    print(e)

