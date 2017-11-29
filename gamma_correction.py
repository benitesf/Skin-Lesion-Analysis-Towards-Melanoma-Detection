from scipy.misc import imread, imsave
from skimage import exposure, img_as_float
import util.dirhandler

"""
Get name images to correction
-----------------------------
"""
melanoma_path = 'image/ISIC-2017_Training_Data_Clean_Preprocessed0/'
melanoma_extension = 'jpg'

all_melanoma = sorted(util.dirhandler.get_file_name_dir(melanoma_path, melanoma_extension))

for img_name in all_melanoma:
    image = imread(melanoma_path + img_name)
    image = img_as_float(image)
    image = exposure.adjust_gamma(image, gamma=2.2)
    imsave('resultados/preprocessed/final_gamma_correction/' + img_name, image)