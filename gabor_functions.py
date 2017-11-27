from skimage.filters import gabor_kernel
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

def fourier(kernel):
    # Fourier Transform
    # Take the fourier transform of the image.
    F1 = fftpack.fft2(kernel.real)
    # Now shift so that low spatial frequencies are in the center.
    F2 = fftpack.fftshift(F1)
    # the 2D power spectrum is:
    psd2D = np.abs(F2)
    return psd2D

f = 0.15
theta = [0, np.pi/4, np.pi/2]
sigma_x = 7
sigma_y = 10
offset = 1/2

#k = gabor_kernel(frequency=f, theta=theta[0], sigma_x=sigma_x, sigma_y=sigma_y)
f_list = []
for t in theta:
    k = gabor_kernel(frequency=f, theta=t)
    f_list.append(fourier(k))

#plt.imshow(fk, cmap='Greys')
plt.imshow(f_list[2])
plt.axis('off')
#plt.gca().invert_yaxis()
plt.show()



