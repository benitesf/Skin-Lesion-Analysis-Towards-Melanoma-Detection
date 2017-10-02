# Import util methods
from scipy.misc import imread, imsave
import util.dirhandler
from preprocessing.contrast_enhancement import median_filter_, shading_attenuation_method
from preprocessing.shades_of_gray import shades_of_gray_method
#import sys, os
#sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
#os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
#from contrast_enhancement import median_filter_
#from shades_of_gray import shades_of_gray_method


"""
Get name images to preprocessing
--------------------------------
"""
melanoma_path = 'image/ISIC-2017_Training_Data_Clean/'
melanoma_extension = 'jpg'

all_melanoma = sorted(util.dirhandler.get_file_name_dir(melanoma_path, melanoma_extension))


"""
Iterate over the list of images
-------------------------------
"""

for img_name in all_melanoma:
    # Get the image
    image = imread(melanoma_path + img_name)

    # Median filter
    image = median_filter_(image)

    # weak effect of nonuniform illumination or shadow
    image = shading_attenuation_method(image, extract=40, margin=10)

    # Color augmentation
    image = shades_of_gray_method(image)
    imsave('image/Preprocessed_Data/'+img_name, image)
