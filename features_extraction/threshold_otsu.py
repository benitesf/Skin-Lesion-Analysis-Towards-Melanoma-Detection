import os
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu


def get_segmentation(image):

    os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection")
    img = io.imread(image)
    img = rgb2gray(img)

    thresh = threshold_otsu(img)
    binary = img <= thresh
    return binary