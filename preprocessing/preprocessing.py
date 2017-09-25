# Import util methods
from math import floor, sqrt
from scipy.misc import imread
from scipy.ndimage.filters import median_filter
from skimage.color import rgb2gray, rgb2hsv, rgb2luv, rgb2lab
from skimage import exposure, img_as_float

import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection")

from util.dirhandler import get_file_name_dir
import config as cfg


def median_filter_(img):
    M, N = img.shape[0:2]
    n = floor(5 * sqrt((M/768) * (N/512)))
    filtered = np.zeros(img.shape, dtype='uint8')

    filtered[:,:,0] = median_filter(img[:,:,0], size=n)
    filtered[:,:,1] = median_filter(img[:,:,1], size=n)
    filtered[:,:,2] = median_filter(img[:,:,2], size=n)

    return filtered


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
Get name images to preprocessing
--------------------------------
"""

melanoma_path = 'image/ISIC-2017_Training_Data_Clean/'
melanoma_extension = 'jpg'

all_melanoma = sorted(get_file_name_dir(melanoma_path, melanoma_extension))

"""
Get image, 3D-array
-------------------
"""
img = imread(melanoma_path + 'ISIC_0000155.jpg')


"""
Median Filter
-------------
"""
filtered = median_filter_(img)


"""
Histogram Equalization
----------------------
"""
# Gray image
img_gray = rgb2gray(filtered)

# Equalization
img_eq = exposure.equalize_hist(img_gray)

# Adaptative Equalization
img_adapteq = exposure.equalize_adapthist(img_gray, clip_limit=0.03)

"""
Print the images
----------------
"""
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(4,6))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_gray, axes[:, 0])
ax_img.set_title('Original image histogram')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 1])
ax_img.set_title('Histogram equalization')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 2])
ax_img.set_title('Adaptive equalization')

plt.subplots_adjust(wspace=0.4)
plt.show()



