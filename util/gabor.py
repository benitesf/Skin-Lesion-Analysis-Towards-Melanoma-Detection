import numpy as np
from skimage.filters import gabor_kernel
from scipy.ndimage.filters import convolve as convolveim


class GaborFilter:
    def __init__(self, frequency, theta, sigma_x=None, sigma_y=None):
        """ Instantiate a Gabor kernel

        Parameters
        ----------
        frequency: float
            Spatial frequency of the harmonic function. Specified in pixels.
        theta: float
            Orientation in radians.
        sigma_x, sigma_y: float, optional
            Standard deviation in x- and y-directions.

        """
        self.frequency = frequency
        self.theta = theta
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.kernel = gabor_kernel(frequency=frequency, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y)

    def convolve_real(self, image):
        """
        Returns a image convolved with the real component of the kernel
        """
        return convolveim(image, np.real(self.kernel), mode='wrap')

    def convolve_imag(self, image):
        """
        Returns a image convolved with the imaginary component of the kernel
        """
        return convolveim(image, np.imag(self.kernel), mode='wrap')

    def magnitude(self, image):
        """
        Returns the magnitude value
        """
        return np.sqrt(self.convolve_real(image) ** 2 + self.convolve_imag(image) ** 2)