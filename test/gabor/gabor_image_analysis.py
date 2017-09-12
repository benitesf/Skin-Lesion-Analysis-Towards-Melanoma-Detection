import numpy as np
import sys, os
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from sklearn import preprocessing
from gabor_filter_banks import gabor_bank


sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection")

path = 'image/image_for_test/'

melanoma = {'00': ('ISIC_0000047.jpg', 'ISIC_0000047_recorte.jpg'),
            '01': ('ISIC_0000049.jpg', 'ISIC_0000049_recorte.jpg'),
            '02': ('ISIC_0000055.jpg', 'ISIC_0000055_recorte.jpg'),
            '03': ('ISIC_0000066.jpg', 'ISIC_0000066_recorte.jpg'),
            '04': ('ISIC_0000077.jpg', 'ISIC_0000077_recorte.jpg')}

dactilar = {'original': 'dactilar.png',
            'recorte0': 'dactilar_recorte_0.png',
            'recorte45': 'dactilar_recorte_45.png',
            'recorte90': 'dactilar_recorte_90.png',
            'recorte180': 'dactilar_recorte_180.png'}

bandas = 'bandas.png'


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
"""from scipy.misc import imread
from scipy import fftpack

img = imread(path + melanoma['00'][0], mode='F')
ground = imread(path + 'ISIC_0000047_segmentation.png', mode='F')
ground /= 255
lesion = img*ground

F1 = fftpack.fft2(lesion)
# Now shift so that low spatial frequencies are in the center.
F2 = fftpack.fftshift(F1)
# the 2D power spectrum is:
psd2D = np.abs(F2)
mms = preprocessing.MinMaxScaler()
filtered = mms.fit_transform(psd2D)"""

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
