import numpy as np
import sys, os
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from sklearn import preprocessing
from gabor_filter_banks import gabor_bank


def plot_image_convolved(images, nrows, ncols, figsize=(14,8)):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    plt.gray()

    fig.suptitle('Imágenes filtradas con el banco de filtros de gabor', fontsize=12)

    for image, ax in zip(images, fig.axes):
        ax.imshow(image, interpolation='nearest')
        ax.axis('off')

    plt.show()


sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")


"""
Genera banco de filtros de gabor
--------------------------------

Definimos los parámetros iniciales que vamos a utilizar para la generación del banco de filtros de Gabor.
A partir de una frecuencia máxima (fmax), un scaling factor (v) y valor de corte de amplitud (b) se van a generar
automáticamente los filtros de Gabor.

Parámetros
----------
fmax: float
    Frecuencia máxima para definir los filtros de Gabor.
ns: scalar
    Número de escalas de kernels que se van a generar.
nd: scalar
    Número de orientación de kernels que se van a generar.
v: float
    Scaling factor, para generar los subsiguientes kernels a partir del primero.
b: float
    Valor de corte de amplitud.

Referencia
----------
https://www.researchgate.net/publication/4214734_Gabor_feature_extraction_for_character_recognition_Comparison_with_gradient_feature
"""

fmax = 0.4
ns = 4
nd = 4
v = 2
b = 1.177

gabor_filter_bank = gabor_bank(fmax=fmax, ns=ns, nd=nd, v=v, b=b)

img = io.imread('image/Random/texture0.jpg')
img_gray = rgb2gray(img)

filtered = []

for gabor in gabor_filter_bank:
    filtered.append(gabor.magnitude(img_gray))

plot_image_convolved(filtered, ns, nd)

"""
--------------------------------
"""