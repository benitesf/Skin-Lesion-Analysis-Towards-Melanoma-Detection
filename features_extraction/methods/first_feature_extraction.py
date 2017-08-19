import numpy as np
from skimage.color import rgb2gray
import time


def features(img, kernels):
    """
    Implements a function to calculate the feature extraction from a block of image

    Parameters
    ----------
    img: 2D array
        The image for calculate its features
    kernels: gabor kernels
        A list of gabor kernels

    Returns:
        Two arrays X and y, with the features. X save the features values and y save the value 0 or 1.
    """
    pix = img.get_random_pixel()
    blk = img.get_block(pix)
    val = values(blk, kernels)
    mel = img.get_ground_pixel(pix)

    return val, mel


def values(blk, kernels):
    rgb = mean_rgb(blk)
    # start_time = time.time()
    gab = gabor_filter(blk, kernels)
    # print("--- %s seconds ---" % (time.time() - start_time))
    return [*rgb, *gab]


def mean_rgb(block):
    """
    Function to calculate a mean rgb from a block of image

    Returns
    -------
        A 1D array with 3 fields
    """
    r = np.mean(block[:, :, 0])
    g = np.mean(block[:, :, 1])
    b = np.mean(block[:, :, 2])
    return [r, g, b]


def gabor_filter(block, kernels):
    """
     Applys the gabor kernels to a block of pixels

    """
    block = rgb2gray(block)
    feats = np.zeros((len(kernels)), dtype=np.double)

    for index, kernel in enumerate(kernels):
        feats[index] = np.mean(kernel.magnitude(block))
    return feats


