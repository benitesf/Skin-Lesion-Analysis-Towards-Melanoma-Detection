from math import pow
from scipy.misc import imread, imsave
from skimage import img_as_float, img_as_int
from skimage import exposure
import numpy as np
import random

import sys, os
sys.path.append("/home/linux1/Escritorio/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
os.chdir("/home/linux1/Escritorio/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")

melanoma_path = 'image/ISIC-2017_Training_Data_Clean/'
melanoma_extension = 'jpg'

img = imread(melanoma_path + 'ISIC_0014797.jpg')
img_new = img_as_float(img)

#gamma = random.gauss(1, 0.1)
img = exposure.adjust_gamma(img, gamma=2.2)

"""
Illuminant estimated using Minkowski norm
-----------------------------------------
"""
p = 6
shape = img.shape
N = shape[0] + shape[1]

Re = np.power(np.power(img_new[:,:,0], p).sum()/N, 1/p)
Ge = np.power(np.power(img_new[:,:,1], p).sum()/N, 1/p)
Be = np.power(np.power(img_new[:,:,2], p).sum()/N, 1/p)

e = np.array([Re, Ge, Be])
e_ = np.sqrt((e**2).sum())
e_gorro = e/e_

d = 1/e_gorro

img_new[:,:,0] = img_new[:,:,0]*d[0]
img_new[:,:,1] = img_new[:,:,1]*d[1]
img_new[:,:,2] = img_new[:,:,2]*d[2]

imsave('Testing14797.jpeg', img_new)

#I = np.identity(3, dtype=float)*e
