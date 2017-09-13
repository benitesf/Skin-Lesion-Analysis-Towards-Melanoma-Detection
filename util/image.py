import numpy as np
import random as rnd
from scipy.misc import imread
from skimage.transform import resize
from sklearn import preprocessing


"""
Implements a class to manage both images melanoma and its ground.
"""


class Image:

    def __init__(self, melanoma, ground=None, block=None):
        """
        Initialize a Image object reading the melanoma and ground images.

        Parameters
        ----------
        melanoma: String
            Path where the melanoma image is saved.
        ground: String, optional
            Path where the ground image is saved. This ground corresponds to the melanoma image
        block: Scalar
            A scalar to define the block dimension to calculate the sample block.
            For block size 3, its consider a 3x3 sample block.
        """
        try:
            mms = preprocessing.MinMaxScaler()
            self.melanoma = imread(melanoma)
            self.portion = int((block-1)/2)
            if ground is not None:
                try:
                    self.ground = imread(ground)
                except:
                    raise AttributeError('Ups.. a wild error appeared. Can not read the ground image.')
        except:
            raise AttributeError('Ups.. a wild error appeared. Can not read the melanoma image.')

    def get_shape(self):
        return self.melanoma.shape

    def get_portion(self):
        return self.portion

    def get_random_pixel(self):
        """
        Calculates a random pixel of the melanoma image which is inside of the boundary and don't exceeds it.

        Return
        ------
        Random pixel which is the central pixel of the sample block.
        """
        size = self.melanoma.shape
        row = [self.portion, size[0] - self.portion - 1]
        col = [self.portion, size[1] - self.portion - 1]

        return np.array([rnd.randint(row[0], row[1]), rnd.randint(col[0], col[1])])

    def get_block(self, pixel):  # Function to read a block dim x dim from the image
        """
        Return a block which central pixel is passed as a parameter
        """
        row = [pixel[0]-self.portion, pixel[0]+self.portion]
        col = [pixel[1]-self.portion, pixel[1] + self.portion]

        return self.melanoma[row[0]:row[1]+1, col[0]:col[1]+1, :]  # Sum 1, because python's issues

    def get_ground_pixel(self, pixel):  # Return the pixel from ground (0 or 1)
        """
        Return the value of a certain ground pixel. This value can be 0 or 1.
        If ground image does not exists, throws an error.
        """
        if self.ground is not None:
            return self.ground[pixel[0], pixel[1]]
        else:
            raise Exception('Ups.. a wild exception appeared. Ground image does not exists.')

    def get_melanoma_pixel(self, pixel):
        """
        Return the value of a certain melanoma pixel.
        """
        return self.melanoma[pixel[0], pixel[1]]

    """
    def extend_image(self, image):
        extention = (self.block_dim * 2) #+ 1
        return resize(image, (image.shape[0] + extention, image.shape[1] + extention),
                      mode='symmetric')  # This methos read a image from a path passed as argument

    def get_original_size(self):
        size = self.image.shape
        return [size[0] - (self.block_dim * 2), size[1] - (self.block_dim * 2)]
    """