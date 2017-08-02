from scipy.spatial import ConvexHull
import numpy as np
import scipy.misc
import config as cfg
import os

def get_theta(image, path):

    if (path):
        # 1.Matrix of points:
        #   We'll be dealing with 2D points so our matrix is 2xm. The 1st row are the x-coordinates,
        #   2nd row are the y-coordinates and m indicates the number of points.
        os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection")
        img = scipy.misc.imread(image, flatten=True)
    else:
        img = image

    y, x = np.nonzero(img)


    # 2.Subtract the mean for each point:
    #   Calculate the mean of the row of x-coordinates. For each x-point, subtract the mean from it. Do the same for
    #   the y-coordinate. Mean subtraction minimises the mean square error of approximating the data and centers the data.

    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x,y])

    # 3.Covariance matrix calculation:
    #   Calculate the 2x2 covariance matrix.

    cov = np.cov(coords)

    # 4.Eigenvector, eigenvalues of covariance matrix:
    #   Find the 2 eigen-pairs of our dataset.

    evals, evecs = np.linalg.eig(cov)

    # 5.Rearrange the eigen-pairs:
    #   Sort by decreasing eigenvalues

    sort_indices = np.argsort(evals)[::-1]
    evec1, evec2 = evecs[:, sort_indices]
    x_v1, y_v1 = evec1

    # Rotation of the tumor
    return  np.arctan((x_v1)/(y_v1))


def get_convex_hull(x, y):
    points = np.array([x, y]).transpose()
    hull = ConvexHull(points)
    return hull

