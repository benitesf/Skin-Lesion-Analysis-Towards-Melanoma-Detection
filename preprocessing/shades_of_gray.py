from skimage import exposure
import numpy as np
import random


def shades_of_gray_method(image):

    #gamma = random.gauss(1, 0.1)
    img = exposure.adjust_gamma(image, gamma=2.2)

    """
    Illuminant estimated using Minkowski norm
    -----------------------------------------
    """
    p = 6
    shape = img.shape
    N = shape[0] + shape[1]

    Re = np.power(np.power(img[:,:,0], p).sum()/N, 1/p)
    Ge = np.power(np.power(img[:,:,1], p).sum()/N, 1/p)
    Be = np.power(np.power(img[:,:,2], p).sum()/N, 1/p)

    e = np.array([Re, Ge, Be])
    e_ = np.sqrt((e**2).sum())
    e_gorro = e/e_

    d = 1/e_gorro

    img[:,:,0] = img[:,:,0]*d[0]
    img[:,:,1] = img[:,:,1]*d[1]
    img[:,:,2] = img[:,:,2]*d[2]

    return img