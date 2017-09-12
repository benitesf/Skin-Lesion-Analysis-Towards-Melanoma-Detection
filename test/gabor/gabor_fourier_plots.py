import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack


def plot_surface3d(Z):
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib import cm
    from mpl_toolkits.mplot3d import axes3d

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.floor(Z.shape[1] / 2).astype(int)
    y = np.floor(Z.shape[0] / 2).astype(int)

    X = np.arange(-x, x + 1, 1)
    Y = np.arange(-y, y + 1, 1)
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=cm.RdBu, linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_facecolor('gray')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_surface2d(Z):
    #plt.imshow(Z, cmap='Greys')
    plt.imshow(Z)
    #plt.gca().invert_yaxis()
    plt.show()
    """
    import pylab as py
    py.figure(1)
    py.clf()
    py.imshow(Z)
    py.show()
    """


def plot_gabor_fourier_2d(kernels, fouriers, nrows, ncols, figsize=(14,8)):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    plt.gray()

    fig.suptitle('Filtros de Gabor con sus respectivas transformadas de Fourier', fontsize=12)

    merge = [None]*(len(kernels)+len(fouriers))
    merge[::2] = kernels
    merge[1::2] = fouriers

    for val, ax in zip(merge, fig.axes):
        ax.imshow(val, interpolation='nearest')
        #ax.invert_yaxis()
        ax.axis('off')

    plt.show()


def plot_sum_gabor_fourier_2d(sum_kernel, sum_fourier):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,4))

    fig.suptitle('Suma de los filtros de Gabor y Fourier', fontsize=10)

    ax = axes[0]
    ax.imshow(sum_kernel)
    ax.axis('off')

    ax = axes[1]
    ax.imshow(sum_fourier)
    ax.axis('off')

    plt.show()


def power(image, kernel):
    from scipy import ndimage as ndi
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap') ** 2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap') ** 2)


def fourier(kernel):
    # Fourier Transform
    # Take the fourier transform of the image.
    F1 = fftpack.fft2(kernel.real)
    # Now shift so that low spatial frequencies are in the center.
    F2 = fftpack.fftshift(F1)
    # the 2D power spectrum is:
    psd2D = np.abs(F2)
    return psd2D

"""
Genera el banco de filtros y su plotting
----------------------------------------
"""
fmax = 1/2
ns = 4
nd = 4
v = 2
b = 1.177

from gabor_filter_banks import gabor_bank

gabor_filter_bank = gabor_bank(fmax, ns, nd)

kernels = []
fouriers = []

# Recoge todos los kernels del banco de filtros
for gabor in gabor_filter_bank:
    kernels.append(gabor.kernel)

# Calcula la transformada de Fourier para cada kernel
for kernel in kernels:
    fouriers.append(fourier(kernel))

kernels_real = []

# Recoge las componentes reales de los kernels
for kernel in kernels:
    kernels_real.append(kernel.real)

#plot_gabor_fourier_2d(kernels_real, fouriers, ns, nd*2)

"""
----------------------------------------
"""


"""
Muestra la suma de todos los filtros de gabor y fourier
-------------------------------------------------------
"""
from skimage.transform import resize

fourier_resize = []

for f in fouriers:
    fourier_resize.append(resize(f, (100,100)))

sum_fourier = np.zeros((100,100))
for val in fourier_resize:
    sum_fourier += val

kernel_resize = []

for k in kernels_real:
    kernel_resize.append(resize(k, (100,100)))

sum_kernel = np.zeros((100,100))
for val in kernel_resize:
    sum_kernel += val

#plot_sum_gabor_fourier_2d(sum_kernel, sum_fourier)

"""
-------------------------------------------------------
"""

