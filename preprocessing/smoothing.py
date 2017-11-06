from scipy.ndimage.filters import median_filter
from math import floor, sqrt
import numpy as np


def median_filter_(img):
    M, N = img.shape[0:2]
    n = floor(5 * sqrt((M/768) * (N/512)))
    filtered = np.zeros(img.shape, dtype='uint8')

    filtered[:,:,0] = median_filter(img[:,:,0], size=n)
    filtered[:,:,1] = median_filter(img[:,:,1], size=n)
    filtered[:,:,2] = median_filter(img[:,:,2], size=n)

    return filtered