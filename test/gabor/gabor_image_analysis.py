import numpy as np
import sys, os
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from sklearn import preprocessing
from gabor_filter_banks import gabor_bank


def plot_surface2d(Z):
    plt.imshow(Z, cmap='Greys')
    #plt.imshow(Z)
    #plt.gca().invert_yaxis()
    plt.show()


def plot_image_convolved(images, nrows, ncols, figsize=(14,8)):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    plt.gray()

    fig.suptitle('Imágenes filtradas con el banco de filtros de gabor', fontsize=12)

    for image, ax in zip(images, fig.axes):
        ax.imshow(image, interpolation='nearest')
        ax.axis('off')

    plt.show()


"""
Análisis de frecuencias de imágenes con lesión
----------------------------------------------
El objetivo es hacer un estudio sobre el espectro de frecuencias en las zonas de lesión.

Para ello, se tomarán las imágenes con melanoma y se aislarán las zonas con lesión y se utilizará la transformada
de Fourier para su estudio.

Este análisis servirá para definir con mayor exactidud la frecuencia del banco de filtros de Gabor.
"""
from scipy.misc import imread
from scipy import fftpack
import sys, os

sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")

melanoma_path = 'image/ISIC-2017_Training_Data_Clean/'
ground_path = 'image/ISIC-2017_Training_Part1_GroundTruth_Clean/'


image = imread(melanoma_path + 'ISIC_0000013.jpg', mode='F')
ground = imread(ground_path + 'ISIC_0000013_segmentation.png', mode='F')
ground /= 255
lesion = image*ground

F1 = fftpack.fft2(lesion)
# Now shift so that low spatial frequencies are in the center.
F2 = fftpack.fftshift(F1)
# the 2D power spectrum is:
psd2D = np.abs(F2)
mms = preprocessing.MinMaxScaler()
filtered = mms.fit_transform(psd2D)
filtered[filtered>0.5] = 1
io.imshow(filtered, cmap='gray')
io.show()
"""
----------------------------------------------
"""


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

"""
fmax = 1/2
ns = 4
nd = 4
v = 2
b = 1.177

gabor_filter_bank = gabor_bank(fmax=fmax, ns=ns, nd=nd, v=v, b=b)

img = io.imread(path + dactilar['original'])
img_gray = rgb2gray(img)

filtered = []

for gabor in gabor_filter_bank:
    filtered.append(gabor.magnitude(img_gray))

plot_image_convolved(filtered, ns, nd)
"""
"""
--------------------------------
"""

"""
Para mejor visualización
------------------------

mms = preprocessing.MinMaxScaler()
filtered = mms.fit_transform(filtered)
plot_surface2d(filtered)
# io.imshow(filtered)
# io.show()
"""
