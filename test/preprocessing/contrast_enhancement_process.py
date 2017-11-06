import sys, os

#sys.path.append("/home/linux1/Escritorio/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
#os.chdir("/home/linux1/Escritorio/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")

from skimage import exposure, img_as_float
from scipy.misc import imread, imsave
import numpy as np

# Paths and filenames
filename = 'ISIC_0000096'

melanoma_path = 'image/ISIC-2017_Training_Data_Clean/'
melanoma_name = filename
melanoma_extension = '.jpg'

pathdir = 'memory/pre-processing/contrast_enhancement/'


def save_image_gamma_correction(X, gamma):
    imsave(pathdir + filename + '_gamma_corrected_' + str(gamma) + melanoma_extension, X)


def save_image_enhanced(X, d, p, g):
    X[:, :, 0] = X[:, :, 0] * d[0]
    X[:, :, 1] = X[:, :, 1] * d[1]
    X[:, :, 2] = X[:, :, 2] * d[2]
    imsave(pathdir + filename + '_enhanced_g' + str(g) + '_p' + str(p) + melanoma_extension, X)


"""
Reading image
-------------
"""
image = imread(melanoma_path + melanoma_name + melanoma_extension)
image = img_as_float(image)

gamma_list = [1.3, 1.5, 1.8, 2.0, 2.2]
p_list = [2, 4, 6]

for g, p in zip(gamma_list, p_list):
    F = exposure.adjust_gamma(image, gamma=g)

    save_image_gamma_correction(F, gamma=g)

    """
    Illuminant estimated using Minkowski norm
    -----------------------------------------
    """
    #p = 6
    shape = F.shape
    N = shape[0] + shape[1]

    Re = np.power(np.power(F[:,:,0], p).sum()/N, 1/p)
    Ge = np.power(np.power(F[:,:,1], p).sum()/N, 1/p)
    Be = np.power(np.power(F[:,:,2], p).sum()/N, 1/p)

    e = np.array([Re, Ge, Be])
    e_ = np.sqrt((e**2).sum())
    e_gorro = e/e_

    d = 1/e_gorro

    save_image_enhanced(F, d, p, g)
