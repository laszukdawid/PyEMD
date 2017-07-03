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

    def __init__(self, **kwargs):
        # Declare constants
        self.std_thr = 0.2
        self.svar_thr = 0.001
        self.power_thr = -5
        self.total_power_thr = 0.01
        self.range_thr = 0.001

        # ProtoIMF related
        self.inst_thr = 0.05
        self.mse_thr = 0.01

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

    def extract_max_min_spline(self, image):

        min_peaks, max_peaks = self.find_extrema(image)

        # Prepare grid for interpolation. Doesn't seem necessary.
        X = np.arange(image.shape[0])
        Y = np.arange(image.shape[1])
        xi, yi = np.meshgrid(X, Y)

        min_env = self.spline_points(min_peaks[0], min_peaks[1], image, xi, yi)
        max_env = self.spline_points(max_peaks[0], max_peaks[1], image, xi, yi)

        return min_env, max_env

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

        min_peaks = (X_min, Y_min) = np.nonzero(min_peaks)
        max_peaks = (X_max, Y_max) = np.nonzero(max_peaks)

	return min_peaks, max_peaks

    def end_condition(self, image, IMFs):
        rec = np.sum(IMFs, axis=0)

        # If reconstruction is perfect, no need for more tests
        if np.allclose(image, rec):
            return True

        return False

    def check_proto_imf(self, proto_imf):

        # No speck above inst_thr
        if np.any(proto_imf > self.inst_thr):
            return False

        # Everything relatively close to 0
        mse_proto_imf = np.mean(proto_imf*proto_imf)
        if mse_proto_imf > self.mse_thr:
            return False

        return True

    def check_imf(self, imf_new, imf_old, eMax, eMin, mean):
        return True

    def emd(self, image, max_imf=-1):

        res = image.copy()
        imf = np.zeros(image.shape)
        imf_olf = imf.copy()

        imfNo = 0
        IMF = np.empty((imfNo,)+imf.shape)
        notFinished = True

        while(notFinished):
            self.logger.debug('IMF -- '+str(imfNo))

            res = image - np.sum(IMF[:imfNo], axis=0)
            imf = res.copy()
            mean = np.zeros(image.shape)

            # Counters
            n = 0   # All iterations for current imf.
            n_h = 0 # counts when mean(proto_imf) < threshold

            while(n<self.MAX_ITERATION):
                n += 1
                self.logger.debug("Iteration: "+str(n))

                min_peaks, max_peaks = self.find_extrema(imf)

                if len(min_peaks)>2 and len(max_peaks):

                    imf_old = imf.copy()
                    imf = imf - mean

                    max_env, min_env, eMax, eMin = self.extract_max_min_spline(T, imf)

                    mean = 0.5*(max_env+min_env)

                    imf_old = imf.copy()
                    imf = imf - mean

                    # Fix number of iterations
                    if self.FIXE:
                        if n>=self.FIXE+1: break

                    # Fix number of iterations after number of zero-crossings
                    # and extrema differ at most by one.
                    elif self.FIXE_H:

                        if n == 1: continue
                        if self.imf_check(imf):
                            n_h += 1
                        else:
                            n_h = 0

                        # STOP if enough n_h
                        if n_h >= self.FIXE_H: break

                    # Stops after default stopping criteria are met
                    else:
                        pass
              #         ext_res = self.find_extrema(T, imf)
              #         max_pos, max_val, min_pos, min_val, ind_zer = ext_res
              #         extNo = len(max_pos) + len(min_pos)
              #         nzm = len(ind_zer)

              #         f1 = self.check_imf(imf, max_env, min_env, mean, extNo)
              #         #f2 = np.all(max_val>0) and np.all(min_val<0)
              #         f2 = abs(extNo - nzm)<2

              #         # STOP
              #         if f1 and f2: break

                else:
                    notFinish = False
                    break

            IMF = np.vstack((IMF, imf.copy()[None,:]))
            imfNo += 1

            if self.end_condition(image, IMF) or imfNo==max_imf:
                notFinish = False
                break

        return IMF

########################################
if __name__ == "__main__":
    x = np.arange(50)
    y = np.arange(50).reshape((-1,1))

    pi2 = 2*np.pi
    img = np.sin(x*pi2)*np.cos(y*3*pi2)+x

    emd2d = EMD2D()
    IMFs = emd2d.emd(img)
    print(IMFs)
    print(IMFs.shape)
