import numpy as np
from skimage.color import rgb2gray, rgb2hsv, rgb2lab, rgb2luv
from scipy.stats import skew
from preprocessing.shadding_attenuation import entropy_ratio, in_range
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
    rgb = rgb_features(blk) # 15
    hsv = hsv_features(blk) # 15
    lab = lab_features(blk) # 15
    luv = luv_features(blk) # 15
    # start_time = time.time()
    gab = gabor_filter(blk, kernels) # 4
    # print("--- %s seconds ---" % (time.time() - start_time))
    return [*rgb, *hsv, *lab, *luv, *gab] # 64


def rgb_features(block):
    return [*mean(block), *std_dev(block), *skew_(block), *energy(block), *entropy(block)]


def hsv_features(block):
    blk = rgb2hsv(block)
    return [*mean(blk), *std_dev(blk), *skew_(blk), *energy(blk), *entropy(blk)]


def lab_features(block):
    blk = rgb2lab(block)
    return [*mean(blk), *std_dev(blk), *skew_(blk), *energy(blk), *entropy(blk)]


def luv_features(block):
    blk = rgb2luv(block)
    return [*mean(blk), *std_dev(blk), *skew_(blk), *energy(blk), *entropy(blk)]


def mean(block):
    """
    Function to calculate the mean of the block

    Returns
    -------
        A 1D array with 3 fields
    """
    a = np.mean(block[:, :, 0])
    b = np.mean(block[:, :, 1])
    c = np.mean(block[:, :, 2])
    return [a, b, c]


def std_dev(block):
    """
    Function to calculate the standard deviation of the block

    Returns
    -------
        A 1D array with 3 fields
    """
    a = np.std(block[:, :, 0])
    b = np.std(block[:, :, 1])
    c = np.std(block[:, :, 2])
    return [a, b ,c]


def skew_(block):
    """
    Function to calculate the skewness of the block

    """
    a = skew(block[:, :, 0].flatten())
    b = skew(block[:, :, 1].flatten())
    c = skew(block[:, :, 2].flatten())
    return [a, b, c]


def energy(block):
    a = np.sqrt((block[:, :, 0] ** 2).sum())
    b = np.sqrt((block[:, :, 1] ** 2).sum())
    c = np.sqrt((block[:, :, 2] ** 2).sum())
    return [a, b, c]


def entropy(block):
    blk = in_range(block)
    a = entropy_ratio(blk[:, :, 0])
    b = entropy_ratio(blk[:, :, 1])
    c = entropy_ratio(blk[:, :, 2])
    return [a, b, c]


def gabor_filter(block, kernels):
    """
     Applys the gabor kernels to a block of pixels

    """
    block = rgb2gray(block)
    feats = np.zeros((len(kernels)), dtype=np.double)

    for index, kernel in enumerate(kernels):
        feats[index] = np.mean(kernel.magnitude(block))
    return feats


