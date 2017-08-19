from skimage.filters import gabor_kernel
from scipy.ndimage.filters import convolve as convolveim
import numpy as np
import time


class GaborFilter:

    real_time = 0
    real_cont = 0

    imag_time = 0
    imag_cont = 0

    magnitude_time = 0
    magnitude_cont = 0

    def __init__(self, frequency, theta, sigma_x, sigma_y):
        """ Instantiate a Gabor kernel

        Parameters
        ----------
        frequency: float
            Spatial frequency of the harmonic function. Specified in pixels.
        theta: float
            Orientation in radians.
        sigma_x, sigma_y: float
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
        start_time = time.time()
        conv = convolveim(image, np.real(self.kernel), mode='wrap')
        self.real_time += time.time() - start_time
        self.real_cont += 1

        return conv

    def convolve_imag(self, image):
        """
        Returns a image convolved with the imaginary component of the kernel
        """
        start_time = time.time()
        conv = convolveim(image, np.imag(self.kernel), mode='wrap')
        self.imag_time += time.time() - start_time
        self.imag_cont += 1

        return conv

    def magnitude(self, image):
        """
        Returns the magnitude value
        """
        start_time = time.time()
        conv = np.sqrt(self.convolve_real(image) ** 2 + self.convolve_imag(image) ** 2)
        self.magnitude_time += time.time() - start_time
        self.magnitude_cont += 1

        return conv

    def mean_magnitude_time(self):
        return self.magnitude_time/self.magnitude_cont

    def mean_real_time(self):
        return self.real_time/self.real_cont

    def mean_imag_time(self):
        return self.imag_time/self.imag_cont


def gabor_bank(fmax, ns, nd, v=2, b=1.177):
    """ Devuelve un array 2D (Ns x Nd) de filtros de Gabor.

    Cada posición dentro del array corresponde a un determinado filtro de Gabor que ha sido creado
    con unos parámetros especificos.

    Parameters
    ----------
    fmax: float
        Max Spatial frequency of the harmonic function. Specified in pixels.
    ns: scalar
        Number of scales in the bank.
    nd: scalar
        Number of orientations in the bank.
    v: float, optional
        Scaling factor to create gabor filters (2 octaves by default)
    b: float, optional
        Value to truncate the Gaussian envelope bandwidth (1.177 by default to truncate at the half of amplitude)

    Returns
    -------
    bank: 2D array
        Bank of Gabor filters

    References
    ----------
    https://www.researchgate.net/publication/4214734_Gabor_feature_extraction_for_character_recognition_Comparison_with_gradient_feature

    """

    # Bank of Gabor Filters
    bank = []

    """
    Initial parameters
    """
    # Interval of orientation
    O = np.pi / nd

    # Aspect ratio
    alpha = np.tan(O / 2) * (v + 1) / (v - 1)

    # Orientations
    thetas = [(i * np.pi / nd) for i in range(0, nd)]

    """
    First nd kerneles with the same frequency but different orientations
    """
    # Frequency
    f = fmax * (v + 1) / (2 * v)

    sigma_u = f * (v - 1) / (b * (v + 1))
    sigma_v = sigma_u / alpha
    sigma_x = 1 / (2 * np.pi * sigma_u)
    sigma_y = 1 / (2 * np.pi * sigma_v)

    for theta in thetas:
        bank.append(GaborFilter(f, theta, sigma_x, sigma_y))

    """
    Rest of kernels with different frequencys and orientations
    """
    for i in range(1, ns):
        f /= v
        sigma_u = f * (v - 1) / (b * (v + 1))
        sigma_v = sigma_u / alpha
        sigma_x = 1 / (2 * np.pi * sigma_u)
        sigma_y = 1 / (2 * np.pi * sigma_v)
        for theta in thetas:
            bank.append(GaborFilter(f, theta, sigma_x, sigma_y))

    return bank
