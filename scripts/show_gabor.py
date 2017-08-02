
import sys, os
sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
import scipy.misc
from features_extraction.majorAxis import get_theta
from skimage.color import rgb2gray, rgb2hsv
from skimage import io
from scipy import ndimage as ndi
import numpy as np
from skimage.filters import gabor_kernel



os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection")
"""
ground_1 = 'ISIC_0000000_segmentation.png'
ground_2 = 'ISIC_0000077_segmentation.png'
ground_3 = 'ISIC_0000156_segmentation.png'

theta_1 = get_theta(ground_1, True)
theta_2 = get_theta(ground_2, True)
theta_3 = get_theta(ground_3, True)

data_path = 'image/train_data/'
data_1 = 'ISIC_0000000.jpg'
data_2 = 'ISIC_0000077.jpg'
data_3 = 'ISIC_0000156.jpg'

image_1 = io.imread(data_path + data_1)
image_2 = io.imread(data_path + data_2)
image_3 = io.imread(data_path + data_3)



img_1 = rgb2hsv(image_1)
#img_2 = rgb2gray(image_2)
#img_3 = rgb2gray(image_3)

"""
frequency = 0.2
bandwidth = 2

kernel_1 = gabor_kernel(frequency, theta=0, bandwidth=bandwidth)
#kernel_2 = gabor_kernel(frequency, theta=theta_2, bandwidth=bandwidth)
#kernel_3 = gabor_kernel(frequency,  bandwidth=bandwidth)

#filtered_1 = ndi.convolve(img_1, kernel_1.real, mode='wrap')
#filtered_2 = ndi.convolve(img_2, kernel_2.real, mode='wrap')
#filtered_3 = ndi.convolve(img_3, kernel_3.real, mode='wrap')




io.imshow(kernel_1.real)
io.show()
