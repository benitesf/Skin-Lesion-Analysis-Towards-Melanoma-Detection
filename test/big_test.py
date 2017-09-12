# Script Name			: main.py
# Author				: Benites Fernandez, Edson
# Created				: 27/02/17
# Last Modified			:
# Version				: 1.0

# Modifications			: 1.1 - some modifications
#						: 1.2 - some modifications

# Description			: Main script to run the system
#

##################################################
##################################################
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
# Get all the image names from train_data and train_ground directory
train_data_set   = sorted(dh.get_file_name_dir(cfg.train_data_path, cfg.data_ext))
train_ground_set = sorted(dh.get_file_name_dir(cfg.train_ground_path, cfg.ground_ext))

# Get all the image names from test_data and test_ground directory
test_data_set   = sorted(dh.get_file_name_dir(cfg.test_data_path, cfg.data_ext))
test_ground_set = sorted(dh.get_file_name_dir(cfg.test_ground_path, cfg.ground_ext))

length = len(train_data_set)
size = cfg.nImage

index_train = np.random.randint(0, length, size)
index_test  = np.random.randint(0, length, size)


train_data   = [train_data_set[i] for i in index_train]
train_ground = [train_ground_set[i] for i in index_train]

test_data = [test_data_set[i] for i in index_test]
test_ground = [test_ground_set[i] for i in index_test]

############################################################################################################
############################################################################################################

feature  = FE.get('MRG') # (MRG or ADV)
learning = LE.get('NN')  # Only NN
classify = Classification(learning, feature)

############################################################################################################
############################################################################################################
print("Building dataset\n")
X_train, y_train = feature.get_data_set(train_data, train_ground, type="train")
#X_test, y_test = feature.get_data_set(test_data, test_ground, type="test")


print("Fitting Neural Network")
learning.fit(X_train, y_train)


print("Classifying")
data = test_data[0:3]
ground = test_ground[0:3]
acc = classify.accurate(data, ground, set='test')