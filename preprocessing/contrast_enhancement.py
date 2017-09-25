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


"""
def plot_img_and_hist(img, axes, bins=256):
    img = img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf
"""