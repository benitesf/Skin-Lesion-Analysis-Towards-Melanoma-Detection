# Import util methods
from scipy.misc import imread, imsave
from skimage import exposure, img_as_float
import sys, os

sys.path.append("/home/linux1/Escritorio/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
os.chdir("/home/linux1/Escritorio/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")

import util.dirhandler
from contrast_enhancement import median_filter_
from shades_of_gray import shades_of_gray_method


"""
Get name images to preprocessing
--------------------------------
"""

melanoma_path = 'image/ISIC-2017_Training_Data_Clean/'
melanoma_extension = 'jpg'

all_melanoma = sorted(util.dirhandler.get_file_name_dir(melanoma_path, melanoma_extension))

"""
Iterate over the list of images
-------------------
"""

for img_name in all_melanoma:
    img = img_as_float(median_filter_(imread(melanoma_path + img_name)))
    img = shades_of_gray_method(img)
    imsave('image/Data_Preprocessed/'+img_name, img)
