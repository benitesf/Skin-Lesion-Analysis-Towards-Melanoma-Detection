from skimage.transform import resize
from scipy.misc import imread, imsave
import numpy as np
import os
import util.dirhandler

melanoma_path = 'image/ISIC-2017_Training_Data_Clean/'
melanoma_extension = 'jpg'

all_melanoma = sorted(util.dirhandler.get_file_name_dir(melanoma_path, melanoma_extension))

info_sizes = [os.stat(melanoma_path+i).st_size/(2**20) for i in all_melanoma]

max_size = all_melanoma[info_sizes.index(max(info_sizes))]

img = imread(melanoma_path+max_size)

shape = np.array(img.shape[0:2])
new_shape = np.floor(shape - (shape*0.15)).astype(int)

r = resize(img, new_shape, mode='reflect')
imsave('image/'+max_size, r)

