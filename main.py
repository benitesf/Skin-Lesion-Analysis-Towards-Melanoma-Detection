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

import classes.dirhandler as dh
import config as cfg
from features_extraction import feature_extraction as FE
from learning import learning as LE
from classification.classification import Classification

##################################################
######### Create training data set ###############
##################################################

# Get all the image names from train_data and train_ground directory
trainDataNames   = sorted(dh.get_file_name_dir(cfg.train_data_path, cfg.data_ext))
trainGroundNames = sorted(dh.get_file_name_dir(cfg.train_ground_path, cfg.ground_ext))

# Instance Features Extraction Object
myFeatureExtraction = FE.get('MRG') # MRG means mean RGB Gabor

# Get the data set [mean r, mean g, mean b, gabor 0, gabor 1, gabor 2, gabor 3, y]
X,y = myFeatureExtraction.getTrainDataSet(trainDataNames, trainGroundNames)

##################################################
######### Training Neural Network ################
##################################################
myLearning = LE.get('NN') ## NN means Neural Network

myLearning.fit(X, y)

##################################################
############## Classify images ###################
##################################################

# Get all the image names from test_data and test_ground directory
test_data_names   = sorted(dh.get_file_name_dir(cfg.test_data_path, cfg.data_ext))
test_ground_names = sorted(dh.get_file_name_dir(cfg.test_ground_path, cfg.ground_ext))

classification = Classification(myLearning, myFeatureExtraction)
#c = classification.classify_test(test_data_names[0])
l = classification.accurate(test_data_names, test_ground_names, set='test')