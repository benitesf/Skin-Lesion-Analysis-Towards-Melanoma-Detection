import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import os
from skimage import io
from skimage.color import rgb2gray

def get_theta(evals, evecs):
    sort_indices = np.argsort(evals)[::-1]
    evec1, evec2 = evecs[:, sort_indices]
    x_v1, y_v1 = evec1
    return np.arctan((x_v1) / (y_v1)), evec1, evec2


"""
This scripts find the axes of an object
"""

"""
First Method: Covariance matrix and eigens
"""
# 1.Matrix of points:
#   We'll be dealing with 2D points so our matrix is 2xm. The 1st row are the x-coordinates,
#   2nd row are the y-coordinates and m indicates the number of points.
os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection")
image_1 = io.imread('image/test_data/ISIC_0012232.jpg')
image_2 = io.imread('image/test_data/ISIC_0012413.jpg')


img1 = rgb2gray(image_1)
img2 = rgb2gray(image_2)

array1 = np.asarray(img1).reshape(-1)
array2 = np.asarray(img2).reshape(-1)


# 2.Subtract the mean for each point:
#   Calculate the mean of the row of x-coordinates. For each x-point, subtract the mean from it. Do the same for
#   the y-coordinate. Mean subtraction minimises the mean square error of approximating the data and centers the data.

array1 = array1 - np.mean(array1)

array2 = array2 - np.mean(array2)

# 3.Covariance matrix calculation:
#   Calculate the 2x2 covariance matrix.
coords = np.vstack([array1,array2])
cov = np.cov(coords)

# 4.Eigenvector, eigenvalues of covariance matrix:
#   Find the 2 eigen-pairs of our dataset.

evals, evecs = np.linalg.eig(cov)

# 5.Rearrange the eigen-pairs:
#   Sort by decreasing eigenvalues

theta, evec1, evec2 = get_theta(evals, evecs)



