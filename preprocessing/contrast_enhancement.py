from scipy.misc import imread
from skimage.restoration import denoise_bilateral
from skimage.transform import estimate_transform

import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection")

melanoma_path = 'image/ISIC-2017_Training_Data_Clean/'
melanoma_extension = 'jpg'

img = imread(melanoma_path + 'ISIC_0000155.jpg')

#img_gamma_0 = exposure.adjust_gamma(img, gamma=0.9)
#img_gamma_1 = exposure.adjust_gamma(img, gamma=1.5)

img_t = estimate_transform()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(4,6))

fig.suptitle('Gamma adjust', fontsize=10)

ax = axes[0]
ax.imshow(img)
ax.axis('off')

ax = axes[1]
ax.imshow(img_bilateral0)
ax.axis('off')

ax = axes[2]
ax.imshow(img_bilateral1)
ax.axis('off')

plt.show()