import sys, os
sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
import util.dirhandler as dh
import config as cfg
from features_extraction import feature_extraction as FE
from learning import learning as LE
from classification.classification import Classification
import numpy as np


############################################################################################################
############################################################################################################
os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection")

train_data_set   = sorted(dh.get_file_name_dir(cfg.train_data_path, cfg.data_ext))
train_ground_set = sorted(dh.get_file_name_dir(cfg.train_ground_path, cfg.ground_ext))

# Imágenes que contienen una zona poblada de bellos
bloom_data   = ('ISIC_0011300.jpg', 'ISIC_0011327.jpg', 'ISIC_0012167.jpg', 'ISIC_0012187.jpg', 'ISIC_0012203.jpg', 'ISIC_0012464.jpg')
bloom_ground = ('ISIC_0011300_segmentation.png', 'ISIC_0011327_segmentation.png', 'ISIC_0012167_segmentation.png',
                'ISIC_0012187_segmentation.png', 'ISIC_0012203_segmentation.png', 'ISIC_0012464_segmentation.png')

# Imágenes con poco contraste
big_low_contrast_data   = ('ISIC_0014289.jpg', 'ISIC_0014079.jpg', 'ISIC_0013965.jpg', 'ISIC_0014850.jpg')
big_low_contrast_ground = ('ISIC_0014289_segmentation.png', 'ISIC_0014079_segmentation.png',
                           'ISIC_0013965_segmentation.png', 'ISIC_0014850_segmentation.png')

small_low_contrast_data   = ('ISIC_0014144.jpg', 'ISIC_0014066.jpg', 'ISIC_0013981.jpg', 'ISIC_0012884.jpg')
small_low_contrast_ground = ('ISIC_0014144_segmentation.png', 'ISIC_0014066_segmentation.png',
                             'ISIC_0013981_segmentation.png', 'ISIC_0012884_segmentation.png')

# Imágenes con mucho contraste
big_high_contrast_data   = ('ISIC_0014216.jpg', 'ISIC_0014606.jpg', 'ISIC_0014593.jpg', 'ISIC_0014341.jpg', 'ISIC_0014782.jpg')
big_high_contrast_ground = ('ISIC_0014216_segmentation.png', 'ISIC_0014606_segmentation.png',
                            'ISIC_0014593_segmentation.png', 'ISIC_0014341_segmentation.png', 'ISIC_0014782_segmentation.png')

small_high_contrast_data   = ('ISIC_0014158.jpg' ,'ISIC_0014093.jpg' ,'ISIC_0014028.jpg' ,'ISIC_0014770.jpg')
small_high_contrast_ground = ('ISIC_0014158_segmentation.png' ,'ISIC_0014093_segmentation.png' ,
                              'ISIC_0014028_segmentation.png' ,'ISIC_0014770_segmentation.png')
############################################################################################################
############################################################################################################

feature  = FE.get('ADV') # (MRG or ADV)
learning = LE.get('NN')  # Only NN
classify = Classification(learning, feature)

############################################################################################################
############################################################################################################
print("Building dataset\n")
X_train, y_train = feature.get_data_set(train_data_set, train_ground_set, type="train")

print("Fitting Neural Network")
learning.fit(X_train, y_train)


print("Classifying bloom images")
acc_bloom = classify.accurate_and_segmentation(bloom_data, bloom_ground, set='test', string='bloom')

print("Classifying big low contrast images")
acc_big_low_contrast = classify.accurate_and_segmentation(big_low_contrast_data, big_low_contrast_ground, set='test', string='big_low_contrast')

print("Classifying small low contrast images")
acc_small_low_contrast = classify.accurate_and_segmentation(small_low_contrast_data, small_low_contrast_ground, set='test', string='small_low_contrast')

print("Classifying big high contrast images")
acc_big_high_contrast = classify.accurate_and_segmentation(big_high_contrast_data, big_high_contrast_ground, set='test', string='big_high_contrast')

print("Classifying small high contrast images")
acc_small_high_contrast = classify.accurate_and_segmentation(small_high_contrast_data, small_high_contrast_ground, set='test', string='small_high_contrast')

f = open('adv_adam.txt', 'w')

f.write("Bloom\n")
for l in acc_bloom:
    f.write(str(l)+"\n")

f.write("\nbig_low_contrast\n")
for l in acc_big_low_contrast:
    f.write(str(l)+"\n")

f.write("\nsmall_low_contrast\n")
for l in acc_small_low_contrast:
    f.write(str(l)+"\n")

f.write("\nbig_high_contrast\n")
for l in acc_big_high_contrast:
    f.write(str(l)+"\n")

f.write("\nsmall_high_contrast\n")
for l in acc_small_high_contrast:
    f.write(str(l)+"\n")

f.close()