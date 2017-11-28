import sys, os

sys.path.append("/home/linux1/Escritorio/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
os.chdir("/home/linux1/Escritorio/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
#sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
#os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")

from skimage import exposure, img_as_float
from scipy.misc import imread, imsave
import numpy as np

# Paths and filenames
filename = 'ISIC_0000198_attenuatedf3'

#melanoma_path = 'image/ISIC-2017_Training_Data_Clean/'
melanoma_path = 'resultados/'
melanoma_name = filename
melanoma_extension = '.png'

#pathdir = 'memory/pre-processing/contrast_enhancement/'
pathdir = 'resultados/'


def save_image_gamma_correction(X, gamma):
    imsave(pathdir + filename + '_gamma_corrected_' + str(gamma) + melanoma_extension, X)


def save_image_enhanced(X, d, p, g):
    E = np.copy(X)
    E[:, :, 0] = E[:, :, 0] * d[0]
    E[:, :, 1] = E[:, :, 1] * d[1]
    E[:, :, 2] = E[:, :, 2] * d[2]
    imsave(pathdir + filename + '_enhanced_g' + str(g) + '_p' + str(p) + melanoma_extension, E)


"""
Reading image
-------------
"""
image = imread(melanoma_path + melanoma_name + melanoma_extension)
image = img_as_float(image)

shape = image.shape
N = shape[0] + shape[1]

gamma = 1/2.2

F = exposure.adjust_gamma(image, gamma=gamma)
save_image_gamma_correction(F, gamma=gamma)

"""
Illuminant estimated using Minkowski norm
-----------------------------------------
"""
p = 6
Re = np.power(np.power(F[:,:,0], p).sum()/N, 1/p)
Ge = np.power(np.power(F[:,:,1], p).sum()/N, 1/p)
Be = np.power(np.power(F[:,:,2], p).sum()/N, 1/p)

e = np.array([Re, Ge, Be])
e_ = np.sqrt((e**2).sum())
e_gorro = e/e_

d = 1/e_gorro

save_image_enhanced(F, d, p, gamma)
