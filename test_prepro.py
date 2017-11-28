from skimage.color import rgb2hsv, hsv2rgb
from matplotlib import pyplot as plt
from scipy.misc import imread
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


def in_range(X):
    min = X.min()
    max = X.max()
    return (X - min) / (max - min)


def retrieve_color(X, muorig):
    mu = X.mean()
    return X*muorig/mu


"""
Fitting polynomial function
---------------------------
"""
melanoma_path = 'image/ISIC-2017_Training_Data_Clean/'
melanoma_extension = 'jpg'

image = imread(melanoma_path + 'ISIC_0000386.jpg')

hsv = rgb2hsv(image)
V = np.copy(hsv[:, :, 2])

extract = 80  # Number of pixel to extract from the corners 20x20
margin = 20  # Margin from the borders

shape = image.shape[0:2]

"""
Sampling pixels
---------------
"""
Yc, Xc = sampling_from_corners(margin=margin, extract=extract, shape=shape)
Yf, Xf = sampling_from_frames(margin=margin, extract=extract, shape=shape)

Zc = np.zeros((Xc.shape))
Zf = np.zeros((Xf.shape))

for j in range(0, Zc.shape[0]):
    Zc[j] = np.copy(V[Yc[j], Xc[j]])

for j in range(0, Zf.shape[0]):
    Zf[j] = np.copy(V[Yf[j], Xf[j]])

"""
Quadratic and cubic polynomial
--------------------
"""
Ac2 = quadratic_polynomial_function(Yc, Xc)
Af2 = quadratic_polynomial_function(Yf, Xf)

Ac3 = cubic_polynomial_function(Yc, Xc)
Af3 = cubic_polynomial_function(Yf, Xf)

"""
Fitting polynomial
------------------
"""
coeffc2 = np.linalg.lstsq(Ac2, Zc)[0]
coefff2 = np.linalg.lstsq(Af2, Zf)[0]

coeffc3 = np.linalg.lstsq(Ac3, Zc)[0]
coefff3 = np.linalg.lstsq(Af3, Zf)[0]

"""
Processed
---------
"""
Vprocc2 = apply_quadratic_function(V, coeffc2)
Vprocf2 = apply_quadratic_function(V, coefff2)
Vprocc3 = apply_cubic_function(V, coeffc3)
Vprocf3 = apply_cubic_function(V, coefff3)

# Convert Value into the range 0-1
Vprocc2 = in_range(Vprocc2)
Vprocf2 = in_range(Vprocf2)
Vprocc3 = in_range(Vprocc3)
Vprocf3 = in_range(Vprocf3)

# ****************************************
# Images without retrieve color

fhsvc2 = np.copy(hsv)
fhsvf2 = np.copy(hsv)
fhsvc3 = np.copy(hsv)
fhsvf3 = np.copy(hsv)

fhsvc2[:, :, 2] = np.copy(Vprocc2)
fhsvf2[:, :, 2] = np.copy(Vprocf2)
fhsvc3[:, :, 2] = np.copy(Vprocc3)
fhsvf3[:, :, 2] = np.copy(Vprocf3)

fattenuatedc2 = hsv2rgb(fhsvc2)
fattenuatedf2 = hsv2rgb(fhsvf2)
fattenuatedc3 = hsv2rgb(fhsvc3)
fattenuatedf3 = hsv2rgb(fhsvf3)

# ****************************************

# Retrieve true color to skin
muorig = V.mean()
Vnewc2 = retrieve_color(Vprocc2, muorig)
Vnewf2 = retrieve_color(Vprocf2, muorig)
Vnewc3 = retrieve_color(Vprocc3, muorig)
Vnewf3 = retrieve_color(Vprocf3, muorig)

# Convert Value into the range 0-1
Vnewc2 = in_range(Vnewc2)
Vnewf2 = in_range(Vnewf2)
Vnewc3 = in_range(Vnewc3)
Vnewf3 = in_range(Vnewf3)

hsvc2 = np.copy(hsv)
hsvf2 = np.copy(hsv)
hsvc3 = np.copy(hsv)
hsvf3 = np.copy(hsv)

hsvc2[:, :, 2] = np.copy(Vnewc2)
hsvf2[:, :, 2] = np.copy(Vnewf2)
hsvc3[:, :, 2] = np.copy(Vnewc3)
hsvf3[:, :, 2] = np.copy(Vnewf3)

attenuatedc2 = hsv2rgb(hsvc2)
attenuatedf2 = hsv2rgb(hsvf2)
attenuatedc3 = hsv2rgb(hsvc3)
attenuatedf3 = hsv2rgb(hsvf3)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(6, 4))

ax = axes[0][0]
ax.imshow(image)
ax.axis('off')
# ****************************
ax = axes[0][1]
ax.imshow(fattenuatedc2)
ax.axis('off')

ax = axes[0][2]
ax.imshow(fattenuatedf2)
ax.axis('off')

ax = axes[0][3]
ax.imshow(fattenuatedc3)
ax.axis('off')

ax = axes[0][4]
ax.imshow(fattenuatedf3)
ax.axis('off')
# ****************************
ax = axes[1]
ax.imshow(attenuatedc2)
ax.axis('off')

ax = axes[2]
ax.imshow(attenuatedf2)
ax.axis('off')

ax = axes[3]
ax.imshow(attenuatedc3)
ax.axis('off')

ax = axes[4]
ax.imshow(attenuatedf3)
ax.axis('off')

plt.show()

"""
Print the corners
-----------------

for y,x in zip(Y, X):
    image[y,x,0] = 255
    image[y,x,1:3] = 0

io.imshow(image)
io.show()
"""
