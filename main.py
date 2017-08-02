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

import util.dirhandler as dh
import config as cfg
from features_extraction import feature_extraction as FE
from learning import learning as LE
from classification.classification import Classification
from sklearn.model_selection import train_test_split
import time

"""
Create training data set
"""
# Get all the image names from train_data and train_ground directory
train_data_set   = sorted(dh.get_file_name_dir(cfg.train_data_path, cfg.data_ext))
train_ground_set = sorted(dh.get_file_name_dir(cfg.train_ground_path, cfg.ground_ext))

# Get all the image names from test_data and test_ground directory
test_data_set   = sorted(dh.get_file_name_dir(cfg.test_data_path, cfg.data_ext))
test_ground_set = sorted(dh.get_file_name_dir(cfg.test_ground_path, cfg.ground_ext))


f = open('lbfgs_200_50_RGB.txt', 'w')
f.write("Accuracy test:\n n_images: %s\tn_sample: %s\n" % (str(cfg.nImage), str(cfg.nSample)))
f.write("frequency: %s\ttheta: %s\n" % (str(cfg.frequency), str(cfg.theta)))


############################################################################################################
############################################################################################################
# Instance Features Extraction Object
myFeatureExtraction = FE.get('MRG') # MRG means mean RGB Gabor

f.write("\nExtracción de características del train: ")
print("Extracción de caracteristicas del train")
start_time = time.time()
# Get the data set [mean r, mean g, mean b, gabor 0, gabor 1, gabor 2, gabor 3, y]
X_train, y_train = myFeatureExtraction.get_data_set(train_data_set, train_ground_set, type="train")
f.write("--- %s seconds ---\n" % (time.time() - start_time))
print("--- %s seconds ---\n" % (time.time() - start_time))

f.write("\nExtracción de características del test: ")
print("Extracción de caracteristicas del test")
start_time = time.time()
# Get the data set [mean r, mean g, mean b, gabor 0, gabor 1, gabor 2, gabor 3, y]
X_test, y_test = myFeatureExtraction.get_data_set(test_data_set, test_ground_set, type="test")
f.write("--- %s seconds ---\n" % (time.time() - start_time))
print("--- %s seconds ---\n" % (time.time() - start_time))
############################################################################################################
############################################################################################################
"""
Training Neural Network
"""
myLearning = LE.get('NN') ## NN means Neural Network

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

print("Aprendizaje")
f.write("\nAprendizaje: ")
start_time = time.time()
myLearning.fit(X_train, y_train)
f.write("\nLoss: %s\n" % myLearning.loss())
f.write("--- %s seconds ---\n" % (time.time() - start_time))
print("--- %s seconds ---\n" % (time.time() - start_time))


score = myLearning.score(X_train, y_train)
f.write("\nScore train set: %s\n" % score)
print("Score train set: %s" % score)

score = myLearning.score(X_test, y_test)
f.write("\nScore test set: %s\n" % score)
print("Score test set: %s "  % score)

############################################################################################################
############################################################################################################
"""
Classify images
"""
f.write("\nClasificación de imágenes train y test. 15 imágenes")

classification = Classification(myLearning, myFeatureExtraction)

train_d = train_data_set[0:15]
train_g = train_ground_set[0:15]

test_d = test_data_set[0:15]
test_g = test_ground_set[0:15]

print("Accurate Train")
f.write("\nAcc train\n")
start_time = time.time()
acc_train = classification.accurate(train_d, train_g, set='train')
f.write("--- %s seconds ---\n" % (time.time() - start_time))
print("--- %s seconds ---\n" % (time.time() - start_time))

for l in acc_train:
    f.write(str(l)+"\n")

print("Accurate Test")
f.write("\nAcc test\n")
start_time = time.time()
acc_test  = classification.accurate(test_d, test_g, set='test')
f.write("--- %s seconds ---\n" % (time.time() - start_time))
print("--- %s seconds ---\n" % (time.time() - start_time))

for l in acc_test:
    f.write(str(l)+"\n")

f.close()
