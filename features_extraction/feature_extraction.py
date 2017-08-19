# First feature extraction method
from features_extraction.methods.first_feature_extraction import features as first_features
from features_extraction.methods.first_feature_extraction import values as first_values

# Second feature extraction method
from features_extraction.methods.second_feature_extraction import features as second_features
from features_extraction.methods.second_feature_extraction import values as second_values

from util.image import Image
from util.gabor import GaborFilter
import numpy as np
import config as cfg

"""
Implements diferent feature extraction methods
----------------------------------------------
"""


class FeatureExtraction:

    def first_method(self, melanoma, ground):
        """
        Implements a first method of feature extraction.
        This method uses some techinques with random parameteres.
        With this method I am going to get some aproximation about how good this working is.

        Parameters
        ----------
        melanoma: list
            A list of strings, which contents is the path of melanoma images
        ground: list
            A list of strings, which contents is a the path of ground images

        Returns
        -------
            A data set X and y, which contents data about the images

        """
        X = np.zeros((cfg.nImage * cfg.nSample, cfg.nCells - 1))
        y = np.zeros((cfg.nImage * cfg.nSample,))

        self.kernels = self.first_gabor_bank(cfg.gabor_params)
        self.values = first_values

        for (melanoma_item, ground_item, image_index) in zip(melanoma, ground, range(cfg.nImage)):
            img = Image(cfg.melanoma_path + melanoma_item, cfg.ground_path + ground_item, cfg.block)
            for sample_index in range(cfg.nSample):
                index = image_index * cfg.nSample + sample_index
                X[index, :], y[index] = first_features(img, self.kernels)
        return X, y.astype(int)

    def second_method(self, melanoma, ground):
        """
        Implements the second method of feature extraction. In this method I consider a better params to create the
        gabor kernels.

        Parameters
        ----------
        melanoma: list
            A list of strings, which contents is the path of melanoma images
        ground: list
            A list of strings, which contents is a the path of ground images

        Returns
        -------
            A data set X and y, which contents data about the images
        """
        X = np.zeros((cfg.nImage * cfg.nSample, cfg.nCells - 1))
        y = np.zeros((cfg.nImage * cfg.nSample,))

        self.kernels = self.second_gabor_bank(fmax=cfg.fmax, ns=cfg.ns, nd=cfg.nd, v=cfg.v, b=cfg.b)
        self.values = second_values

        for (melanoma_item, ground_item, image_index) in zip(melanoma, ground, range(cfg.nImage)):
            img = Image(cfg.melanoma_path + melanoma_item, cfg.ground_path + ground_item, cfg.block)
            for sample_index in range(cfg.nSample):
                index = image_index * cfg.nSample + sample_index
                X[index, :], y[index] = second_features(img, self.kernels)
        return X, y.astype(int)

    def features(self, blk):
        return self.values(blk, self.kernels)

    def get_kernels_bank(self):
        return self.kernels

    def first_gabor_bank(self, params):
        kernels = []
        for frequency in params[0]:
            for theta in params[1]:
                kernels.append(GaborFilter(frequency=frequency, theta=theta))
        return kernels

    def second_gabor_bank(self, fmax, ns, nd, v=2, b=1.177):
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


"""
----------------------------------
"""
