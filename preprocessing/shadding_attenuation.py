from skimage.filters.rank import entropy
from skimage import img_as_ubyte
from math import log2
import numpy as np


def sampling_from_corners(margin, extract, shape):
    """
    Sampling from four corners

    Parameters
    ----------
    margin: scalar
        A margin from the corners
    extract: scalar
        Size of the block sampling, extract x extract
    shape: [a,b] list
        Shape of the image to sampling

    Returns
    -------
    A X and Y array 1D with pixel index
    """

    # Left-top corner
    y = list(range(margin, margin + extract))
    x = list(range(margin, margin + extract))

    Xlt, Ylt = np.meshgrid(x, y, copy=False)

    # Right-top corner
    y = list(range(margin, margin + extract))
    x = list(range(shape[1] - margin - extract, shape[1] - margin))

    Xrt, Yrt = np.meshgrid(x, y, copy=False)

    # Left-bottom corner
    y = list(range(shape[0] - margin - extract, shape[0] - margin))
    x = list(range(margin, margin + extract))

    Xlb, Ylb = np.meshgrid(x, y, copy=False)

    # Right-bottom corner
    y = list(range(shape[0] - margin - extract, shape[0] - margin))
    x = list(range(shape[1] - margin - extract, shape[1] - margin))

    Xrb, Yrb = np.meshgrid(x, y, copy=False)

    X = np.vstack((Xlt, Xrt, Xlb, Xrb)).flatten()
    Y = np.vstack((Ylt, Yrt, Ylb, Yrb)).flatten()

    return Y, X


def sampling_from_frames(margin, extract, shape):
    """
    Sampling from the frames

    Parameters
    ----------
    margin: scalar
        A margin from the corners
    extract: scalar
        Size of the frame sampling
    shape: [a,b] list
        Shape of the image to sampling

    Returns
    -------
    A X and Y array 1D with pixel index
    """

    # Top frame
    y = list(range(margin, margin + extract))
    x = list(range(margin, shape[1] - margin))

    Xt, Yt = np.meshgrid(x, y, copy=False)

    # Right frame
    y = list(range(margin + extract, shape[0] - margin - extract))
    x = list(range(shape[1] - margin - extract, shape[1] - margin))

    Xr, Yr = np.meshgrid(x, y, copy=False)

    # Bottom frame
    y = list(range(shape[0] - margin - extract, shape[0] - margin))
    x = list(range(margin, shape[1] - margin))

    Xb, Yb = np.meshgrid(x, y, copy=False)

    # Left frame
    y = list(range(margin + extract, shape[0] - margin - extract))
    x = list(range(margin, margin + extract))

    Xl, Yl = np.meshgrid(x, y, copy=False)

    X = np.concatenate((Xt.flatten(), Xr.flatten(), Xb.flatten(), Xl.flatten()))
    Y = np.concatenate((Yt.flatten(), Yr.flatten(), Yb.flatten(), Yl.flatten()))

    return Y, X


def quadratic_polynomial_function(Y, X):
    return np.array([X ** 2, Y ** 2, X * Y, X, Y, X * 0 + 1]).T


def cubic_polynomial_function(Y, X):
    return np.array([X ** 3, Y ** 3, (X ** 2) * Y, X * (Y ** 2), X ** 2, Y ** 2, X * Y, X, Y, X * 0 + 1]).T


def apply_quadratic_function(V, coeff):
    Vproc = np.copy(V)
    shape = Vproc.shape
    for y in range(0, shape[0]):
        for x in range(0, shape[1]):
            Vproc[y, x] /= coeff[0] * (x ** 2) + coeff[1] * (y ** 2) + coeff[2] * x * y + coeff[3] * x + coeff[4] * y + \
                       coeff[5]
    return Vproc


def apply_cubic_function(V, coeff):
    Vproc = np.copy(V)
    shape = Vproc.shape
    for y in range(0, shape[0]):
        for x in range(0, shape[1]):
            Vproc[y, x] /= coeff[0] * (x ** 3) + coeff[1] * (y ** 3) + coeff[2] * (x ** 2) * y + coeff[3] * x * (y ** 2) + \
                       coeff[4] * (x ** 2) + coeff[5] * (y ** 2) + coeff[6] * x * y + coeff[7] * x + coeff[8] * y + coeff[9]
    return Vproc


def entropy_ratio(src):
    img = img_as_ubyte(src)
    den, bins = np.histogram(img, bins=range(255), density=True)
    H = 0
    for i in range(den.shape[0]):
        if den[i] > 0:
            H += den[i]*log2(den[i])
    return H*-1


def in_range(X):
    min = X.min()
    max = X.max()
    if (max - min) == 0:
        return np.ones(X.shape)
    return (X - min) / (max - min)


def retrieve_color(X, muorig):
    mu = X.mean()
    return X*muorig/mu