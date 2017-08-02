import matplotlib.pyplot as plt
import os
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection")
data_path = 'image/test_data/'
data_1 = 'ISIC_0013084.jpg'
image_1 = io.imread(data_path + data_1)
img_1 = rgb2gray(image_1)

thresh = threshold_otsu(img_1)
binary = img_1 <= thresh

plt.imshow(binary)
plt.show()