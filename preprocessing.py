# Import util methods
from scipy.misc import imread, imsave
import util.dirhandler
from preprocessing.smoothing import median_filter_
from preprocessing.illumination_enhancement import shading_attenuation_method
from preprocessing.contrast_enhancement import shades_of_gray_method
import os
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
    print('Preprocessing: '+img_name)
    # Get the image
    image = imread(melanoma_path + img_name)

    print('\tfiltering...')
    # Median filter
    image = median_filter_(image)

    print('\tweaking shadows...')
    # weak effect of nonuniform illumination or shadow
    image = shading_attenuation_method(image, extract=50, margin=10)

    print('\tcolor normalization...')
    # Color normalization
    image = shades_of_gray_method(image, gamma=2.2)
    imsave('resultados/preprocessed/' + img_name, image)
