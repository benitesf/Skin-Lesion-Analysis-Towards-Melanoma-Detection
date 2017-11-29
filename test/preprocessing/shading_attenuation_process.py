import sys, os

#sys.path.append("/home/linux1/Escritorio/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
#os.chdir("/home/linux1/Escritorio/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")

from skimage.color import rgb2hsv, hsv2rgb
from skimage import img_as_ubyte
from scipy.misc import imread, imsave
from preprocessing import shadding_attenuation as shatt
from sklearn.metrics.cluster import entropy

import matplotlib
matplotlib.use('Agg') # Trick to not use _Tkinter
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

import numpy as np

# Paths and filenames
melanoma_path = 'image/ISIC-2017_Training_Data_Clean/'
melanoma_name = 'ISIC_0014762'
melanoma_extension = '.jpg'

groundtruth_path = 'image/ISIC-2017_Training_Part1_GroundTruth_Clean/'
groundtruth_name = 'ISIC_0014762_segmentation'
groundtruth_extension = '.png'

#pathdir = "memory/pre-processing/illumination_enhancement/"
pathdir = "resultados/"


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


def plot_surface3d(Y, X, Z, pathdir, filename):
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
    plt.savefig(pathdir + filename)


def save_sampling_pixels(groundtruth):
    gtc = np.copy(groundtruth)
    gtf = np.copy(groundtruth)

    for y, x in zip(Yc, Xc):
        gtc[y, x] = 150

    for y, x in zip(Yf, Xf):
        gtf[y, x] = 150

    imsave(pathdir + groundtruth_name + '_corner_sampled' + groundtruth_extension, gtc)
    imsave(pathdir + groundtruth_name + '_frame_sampled' + groundtruth_extension, gtf)


def save_polynomial_figure(coeffc2, coefff2, coeffc3, coefff3, shape):
    Y = np.arange(0, shape[0], 1)
    X = np.arange(0, shape[1], 1)
    Y, X = np.meshgrid(X, Y)
    Z = np.zeros([shape[0], shape[1]])

    quadratic_polynomial(Z, coeffc2)
    plot_surface3d(Y, X, Z, pathdir=pathdir, filename=melanoma_name + "_quadratic_corner")
    quadratic_polynomial(Z, coefff2)
    plot_surface3d(Y, X, Z, pathdir=pathdir, filename=melanoma_name + "_quadratic_frame")

    cubic_polynomial(Z, coeffc3)
    plot_surface3d(Y, X, Z, pathdir=pathdir, filename=melanoma_name + "_cubic_corner")
    cubic_polynomial(Z, coefff3)
    plot_surface3d(Y, X, Z, pathdir=pathdir, filename=melanoma_name + "_cubic_frame")


def save_without_retrieve_color(hsv, Vprocc2, Vprocf2, Vprocc3, Vprocf3):
    hsvc2 = np.copy(hsv)
    hsvf2 = np.copy(hsv)
    hsvc3 = np.copy(hsv)
    hsvf3 = np.copy(hsv)

    hsvc2[:, :, 2] = np.copy(Vprocc2)
    hsvf2[:, :, 2] = np.copy(Vprocf2)
    hsvc3[:, :, 2] = np.copy(Vprocc3)
    hsvf3[:, :, 2] = np.copy(Vprocf3)

    attenuatedc2 = hsv2rgb(hsvc2)
    attenuatedf2 = hsv2rgb(hsvf2)
    attenuatedc3 = hsv2rgb(hsvc3)
    attenuatedf3 = hsv2rgb(hsvf3)

    imsave(pathdir + melanoma_name + '_attenuatedc2_worc.png', attenuatedc2)
    imsave(pathdir + melanoma_name + '_attenuatedf2_worc.png', attenuatedf2)
    imsave(pathdir + melanoma_name + '_attenuatedc3_worc.png', attenuatedc3)
    imsave(pathdir + melanoma_name + '_attenuatedf3_worc.png', attenuatedf3)

    imsave(pathdir + melanoma_name + '_Vprocc2.png', Vprocc2)
    imsave(pathdir + melanoma_name + '_Vprocf2.png', Vprocf2)
    imsave(pathdir + melanoma_name + '_Vprocc3.png', Vprocc3)
    imsave(pathdir + melanoma_name + '_Vprocf3.png', Vprocf3)


def save_with_retrieve_color(hsv, Vnewc2, Vnewf2, Vnewc3, Vnewf3):
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

    imsave(pathdir + melanoma_name + '_attenuatedc2.png', attenuatedc2)
    imsave(pathdir + melanoma_name + '_attenuatedf2.png', attenuatedf2)
    imsave(pathdir + melanoma_name + '_attenuatedc3.png', attenuatedc3)
    imsave(pathdir + melanoma_name + '_attenuatedf3.png', attenuatedf3)

    imsave(pathdir + melanoma_name + '_Vnewc2.png', Vnewc2)
    imsave(pathdir + melanoma_name + '_Vnewf2.png', Vnewf2)
    imsave(pathdir + melanoma_name + '_Vnewc3.png', Vnewc3)
    imsave(pathdir + melanoma_name + '_Vnewf3.png', Vnewf3)

"""
Reading image
---------------------------
"""
image = imread(melanoma_path + melanoma_name + melanoma_extension)
gt = imread(groundtruth_path + groundtruth_name + groundtruth_extension)

hsv = rgb2hsv(image)
V = np.copy(hsv[:, :, 2])

imsave(pathdir + melanoma_name + '_V.png', V)

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
Saving sampling pixels
----------------------
"""
save_sampling_pixels(gt)

sys.exit()
"""
Quadratic and cubic polynomial
------------------------------
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
save_polynomial_figure(coeffc2, coefff2, coeffc3, coefff3, shape)

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

#save_without_retrieve_color(hsv, Vprocc2, Vprocf2, Vprocc3, Vprocf3)

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

save_with_retrieve_color(hsv, Vnewc2, Vnewf2, Vnewc3, Vnewf3)

# Select the image which have least entropy
Vlist = [V, Vnewc2, Vnewf2, Vnewc3, Vnewf3]
values = [img_as_ubyte(v) for v in Vlist]

entropy_vals = [entropy(v) for v in values]

print('entropy: '+str(entropy_vals))
print('index: '+str(entropy_vals.index(min(entropy_vals))))

