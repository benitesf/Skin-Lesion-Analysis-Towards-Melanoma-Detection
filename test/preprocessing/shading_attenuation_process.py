import sys, os

sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")

from skimage.color import rgb2hsv, hsv2rgb
from scipy.misc import imread, imsave
from preprocessing import shadding_attenuation as shatt

import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

import numpy as np


def quadratic_polynomial(Z, coeff):
    shape = Z.shape
    for y in range(0, shape[0]):
        for x in range(0, shape[1]):
            Z[y, x] = coeff[0] * (x ** 2) + coeff[1] * (y ** 2) + coeff[2] * x * y + coeff[3] * x + \
                      coeff[4] * y + coeff[5]


def cubic_polynomial(Z, coeff):
    shape = Z.shape
    for y in range(0, shape[0]):
        for x in range(0, shape[1]):
            Z[y, x] = coeff[0] * (x ** 3) + coeff[1] * (y ** 3) + coeff[2] * (x ** 2) * y + coeff[3] * x * (y ** 2) + \
                       coeff[4] * (x ** 2) + coeff[5] * (y ** 2) + coeff[6] * x * y + coeff[7] * x + coeff[8] * y + coeff[9]


def plot_surface3d(Y, X, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(Y, X, Z, rstride=2, cstride=2, cmap=cm.RdBu, linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_facecolor('gray')
    ax.invert_xaxis()
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    #plt.show()
    plt.savefig('quadratic')

"""
Reading image
---------------------------
"""
melanoma_path = 'image/ISIC-2017_Training_Data_Clean/'
melanoma_name = 'ISIC_0000246'
melanoma_extension = '.jpg'

groundtruth_path = 'image/ISIC-2017_Training_Part1_GroundTruth_Clean/'
groundtruth_name = 'ISIC_0000246_segmentation'
groundtruth_extension = '.png'

image = imread(melanoma_path + melanoma_name + melanoma_extension)
gt = imread(groundtruth_path + groundtruth_name + groundtruth_extension)

hsv = rgb2hsv(image)
V = np.copy(hsv[:, :, 2])
# imsave('memory/pre-processing/illumination_enhacement/'+melanoma_name+'_V.png', V)

extract = 50  # Number of pixel to extract from the corners 20x20
margin = 10  # Margin from the borders

shape = image.shape[0:2]

"""
Sampling pixels
---------------
"""
Yc, Xc = shatt.sampling_from_corners(margin=margin, extract=extract, shape=shape)
Yf, Xf = shatt.sampling_from_frames(margin=margin, extract=extract, shape=shape)

Zc = np.zeros((Xc.shape))
Zf = np.zeros((Xf.shape))

for j in range(0, Zc.shape[0]):
    Zc[j] = np.copy(V[Yc[j], Xc[j]])

for j in range(0, Zf.shape[0]):
    Zf[j] = np.copy(V[Yf[j], Xf[j]])

"""
Printing sampling pixels
------------------------

gtc = np.copy(gt)
gtf = np.copy(gt)

for y,x in zip(Yc, Xc):
    gtc[y, x] = 150

for y,x in zip(Yf, Xf):
    gtf[y, x] = 150

imsave('memory/pre-processing/illumination_enhacement/'+groundtruth_name+'_corner_sampled'+groundtruth_extension, gtc)
imsave('memory/pre-processing/illumination_enhacement/'+groundtruth_name+'_frame_sampled'+groundtruth_extension, gtf)
"""
"""
Quadratic and cubic polynomial
--------------------
"""
Ac2 = shatt.quadratic_polynomial_function(Yc, Xc)
Af2 = shatt.quadratic_polynomial_function(Yf, Xf)

Ac3 = shatt.cubic_polynomial_function(Yc, Xc)
Af3 = shatt.cubic_polynomial_function(Yf, Xf)

"""
Fitting polynomial
------------------
"""
coeffc2 = np.linalg.lstsq(Ac2, Zc)[0]
coefff2 = np.linalg.lstsq(Af2, Zf)[0]

coeffc3 = np.linalg.lstsq(Ac3, Zc)[0]
coefff3 = np.linalg.lstsq(Af3, Zf)[0]

"""
Saving quadratic and cubic polynomial figure
--------------------------------------------
"""
Y = np.arange(0, shape[0], 1)
X = np.arange(0, shape[1], 1)
Y, X = np.meshgrid(X, Y)
Z = np.zeros([shape[0], shape[1]])
quadratic_polynomial(Z, coeffc2)

plot_surface3d(Y, X, Z)


sys.exit()
"""
Processed
---------
"""
Vprocc2 = shatt.apply_quadratic_function(V, coeffc2)
Vprocf2 = shatt.apply_quadratic_function(V, coefff2)
Vprocc3 = shatt.apply_cubic_function(V, coeffc3)
Vprocf3 = shatt.apply_cubic_function(V, coefff3)

# Convert Value into the range 0-1
Vprocc2 = shatt.in_range(Vprocc2)
Vprocf2 = shatt.in_range(Vprocf2)
Vprocc3 = shatt.in_range(Vprocc3)
Vprocf3 = shatt.in_range(Vprocf3)

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
Vnewc2 = shatt.retrieve_color(Vprocc2, muorig)
Vnewf2 = shatt.retrieve_color(Vprocf2, muorig)
Vnewc3 = shatt.retrieve_color(Vprocc3, muorig)
Vnewf3 = shatt.retrieve_color(Vprocf3, muorig)

# Convert Value into the range 0-1
Vnewc2 = shatt.in_range(Vnewc2)
Vnewf2 = shatt.in_range(Vnewf2)
Vnewc3 = shatt.in_range(Vnewc3)
Vnewf3 = shatt.in_range(Vnewf3)

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
