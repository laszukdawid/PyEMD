#!/usr/bin/python
# coding: UTF-8
#
# Author:   Dawid Laszuk
# Contact:  laszukdawid@gmail.com
#
# Feel free to contact for any information.

from __future__ import division, print_function

import logging
import numpy as np

from scipy.interpolate import Rbf
from skimage.measure import find_contours
from skimage.morphology import reconstruction
from skimage.morphology import binary_dilation
from skimage.morphology import binary_erosion

class BEMD:
    """
    **Bidimensional Empirical Mode Decomposition**

    Method decomposes 2D arrays like gray-scale images into 2D representations of
    Intrinsic Mode Functions (IMFs).

    The algorithm is based on Nunes et. al. [1] work.

    [1] J.-C. Nunes, Y. Bouaoune, E. Delechelle, O. Niang, P. Bunel.,
    "Image analysis by bidimensional empirical mode decomposition. Image and Vision Computing",
    Elsevier, 2003, 21 (12), pp.1019-1026.
    """

    logger = logging.getLogger(__name__)

    def __init__(self):
        # ProtoIMF related
        self.mse_thr = 0.01
        self.mean_thr = 0.01

        self.std_dev = 0.05 # 0.05--0.75 [1]
        self.FIXE = 0
        self.FIXE_H = 0

        self.MAX_ITERATION = 1000

    def __call__(self, image, max_imf=-1):
        return self.bemd(image, max_imf=max_imf)

    def extract_max_min_spline(self, image):
        """Calculates top and bottom envelopes for image.

        Parameters
        ----------
        image : numpy 2D array

        Returns
        -------
        min_env : numpy 2D array
            Bottom envelope in form of an image.
        max_env : numpy 2D array
            Top envelope in form of an image.
        """
        # Prepare grid for interpolation
        xi = np.arange(image.shape[0],image.shape[0]*2)
        yi = np.arange(image.shape[1],image.shape[1]*2)
        min_env = self.spline_points(min_peaks[0], min_peaks[1], min_val, xi, yi)
        max_env = self.spline_points(max_peaks[0], max_peaks[1], max_val, xi, yi)
        return min_env, max_env

    @classmethod
    def spline_points(cls, X, Y, Z, xi, yi):
        """Interpolates for given set of points"""
        spline = Rbf(X, Y, Z, function='linear')
        return spline(xi, yi)

    @classmethod
    def find_extrema(cls, image):
        """
        Finds extrema, both mininma and maxima, based on morphological reconstruction.
        Returns extrema where the first and second elements are x and y positions, respectively.

        Parameters
        ----------
        image : numpy 2D array
            Monochromatic image or any 2D array.

        Returns
        -------
        min_peaks : numpy array
            Minima positions.
        max_peaks : numpy array
            Maxima positions.
        """

        # Extract local extrema
        min_peaks = BEMD.extract_minima(image)
        max_peaks = BEMD.extract_maxima(image)

        return min_peaks, max_peaks

    @classmethod
    def extract_minima(cls, image):
        n = 4
        x_cm = np.random.randint(0, image.shape[1], 4)
        y_cm = np.random.randint(0, image.shape[0], 4)
        min_peaks_pos = np.array([x_cm, y_cm])
        return min_peaks_pos

    @classmethod
    def extract_maxima(cls, image):
        seed_min = image - 1
        dilated = reconstruction(seed_min, image, method='dilation')
        cleaned_image = image - dilated
        return np.where(cleaned_image!=0)

    @classmethod
    def end_condition(cls, image, IMFs):
        """Determins whether decomposition should be stopped.

        Parameters
        ----------
        image : numpy 2D array
            Input image which is decomposed.
        IMFs : numpy 3D array
            Array for which first dimensions relates to respective IMF,
            i.e. (numIMFs, imageX, imageY).
        """
        rec = np.sum(IMFs, axis=0)

        # If reconstruction is perfect, no need for more tests
        if np.allclose(image, rec):
            return True

        return False

    def check_proto_imf(self, proto_imf, proto_imf_prev, mean_env):
        """Check whether passed (proto) IMF is actual IMF.
        Current condition is solely based on checking whether the mean is below threshold.

        Parameters
        ----------
        proto_imf : numpy 2D array
            Current iteration of proto IMF.
        proto_imf_prev : numpy 2D array
            Previous iteration of proto IMF.
        mean_env : numpy 2D array
            Local mean computed from top and bottom envelopes.

        Returns
        -------
        boolean
            Whether current proto IMF is actual IMF.
        """


        #TODO: Sifiting is very sensitive and subtracting const val can often flip
        #      maxima with minima in decompoisition and thus repeating above/below
        #      behaviour. For now, mean_env is checked whether close to zero excluding
        #      its offset.
        if np.all(np.abs(mean_env-mean_env.mean())<self.mean_thr):
        #if np.all(np.abs(mean_env)<self.mean_thr):
            return True

        # If very little change with sifting
        if np.allclose(proto_imf, proto_imf_prev):
            return True

        # If IMF mean close to zero (below threshold)
        if np.mean(np.abs(proto_imf))<self.mean_thr:
            return True

        # Everything relatively close to 0
        mse_proto_imf = np.mean(proto_imf*proto_imf)
        if mse_proto_imf > self.mse_thr:
            return False

        return False

    def bemd(self, image, max_imf=-1):
        """Performs bidimensional EMD (BEMD) on grey-scale image with specified parameters.

        Parameters
        ----------
        image : numpy 2D array,
            Grey-scale image.
        max_imf : int, (default: -1)
            IMF number to which decomposition should be performed.
            Negative value means *all*.

        Returns
        -------
        IMFs : numpy 3D array
            Set of IMFs in form of numpy array where the first dimension
            relates to IMF's ordinary number.
        """
        image_min, image_max = np.min(image), np.max(image)
        offset = image_min
        scale = image_max-image_min

        image_s = (image-offset)/scale

        imf = np.zeros(image.shape)
        imf_old = imf.copy()

        imfNo = 0
        IMF = np.empty((imfNo,)+image.shape)
        notFinished = True

        while(notFinished):
            self.logger.debug('IMF -- '+str(imfNo))

            res = image_s - np.sum(IMF[:imfNo], axis=0)
            imf = res.copy()
            mean_env = np.zeros(image.shape)
            stop_sifting = False

            # Counters
            n = 0   # All iterations for current imf.
            n_h = 0 # counts when mean(proto_imf) < threshold

            while(not stop_sifting and n<self.MAX_ITERATION):
                n += 1
                self.logger.debug("Iteration: "+str(n))

                min_peaks, max_peaks = self.find_extrema(imf)

                self.logger.debug("min_peaks = %i  |  max_peaks = %i" %(len(min_peaks[0]), len(max_peaks[0])))
                if len(min_peaks[0])>4 and len(max_peaks[0])>4:

                    imf_old = imf.copy()
                    imf = imf - mean_env

                    min_env, max_env = self.extract_max_min_spline(imf)

                    mean_env = 0.5*(min_env+max_env)

                    imf_old = imf.copy()
                    imf = imf - mean_env

                    # Fix number of iterations
                    if self.FIXE:
                        if n>=self.FIXE+1:
                            stop_sifting = True

                    # Fix number of iterations after number of zero-crossings
                    # and extrema differ at most by one.
                    elif self.FIXE_H:

                        if n == 1: continue
                        if self.check_proto_imf(imf, imf_old, mean_env):
                            n_h += 1
                        else:
                            n_h = 0

                        # STOP if enough n_h
                        if n_h >= self.FIXE_H:
                            stop_sifting = True

                    # Stops after default stopping criteria are met
                    else:

                        if self.check_proto_imf(imf, imf_old, mean_env):
                            stop_sifting = True

                else:
                    notFinished = False
                    stop_sifting = True

            IMF = np.vstack((IMF, imf.copy()[None,:]))
            imfNo += 1

            if self.end_condition(image, IMF) or imfNo>=max_imf:
                notFinished = False
                break

        res = image_s - np.sum(IMF[:imfNo], axis=0)
        if not np.allclose(res, 0):
            IMF = np.vstack((IMF, res[None,:]))
            imfNo += 1

        IMF = IMF*scale
        IMF[-1] += offset
        return IMF

########################################
if __name__ == "__main__":
    print("Running example on BEMD")
    PLOT = True

    logging.basicConfig(level=logging.DEBUG)

    # Generate image
    print("Generating image... ", end="")
    rows, cols = 1024, 1024
    row_scale, col_scale = 256, 256
    x = np.arange(rows)/float(row_scale)
    y = np.arange(cols).reshape((-1,1))/float(col_scale)

    pi2 = 2*np.pi
    img = np.zeros((rows,cols))
    img = img + np.sin(2*pi2*x)*np.cos(y*4*pi2+4*x*pi2)
    img = img + 3*np.sin(2*pi2*x)+2
    img = img + 5*x*y + 2*(y-0.2)*y
    print("Done")

    # Perform decomposition
    print("Performing decomposition... ", end="")
    emd2d = EMD2D()
    #emd2d.FIXE_H = 5
    IMFs = emd2d.emd(img, max_imf=4)
    imfNo = IMFs.shape[0]
    print("Done")

    if PLOT:
        print("Plotting results... ", end="")
        import pylab as plt

        # Save image for preview
        plt.figure(figsize=(4,4*(imfNo+1)))
        plt.subplot(imfNo+1, 1, 1)
        plt.imshow(img)
        plt.colorbar()
        plt.title("Input image")

        # Save reconstruction
        for n, imf in enumerate(IMFs):
            plt.subplot(imfNo+1, 1, n+2)
            plt.imshow(imf)
            plt.colorbar()
            plt.title("IMF %i"%(n+1))

        plt.savefig("image_decomp")
        print("Done")
