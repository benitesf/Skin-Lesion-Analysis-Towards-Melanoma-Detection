# Import methods of features extraction
from features_extraction.feature_extraction import FeatureExtraction

# Import methods of learning
from learning.learning import neural_network

# Import methods of classification
from classification.classification import classify, confusion_matrix, total_error, local_error

#
from skimage import io
from PIL import Image

# Import util methods
from sklearn.model_selection import train_test_split
import util.dirhandler as dh
import config as cfg
import numpy as np
import time
import sys

"""
Get train and test set
----------------------
"""

all_melanoma = sorted(dh.get_file_name_dir(cfg.melanoma_path, cfg.melanoma_extension))
all_ground = sorted(dh.get_file_name_dir(cfg.ground_path, cfg.ground_extension))

melanoma_train, melanoma_test, ground_train, ground_test = train_test_split(all_melanoma, all_ground, test_size=0.25,
                                                                            random_state=25)


"""
----------------------
"""

"""
Feature Extraction
------------------
"""
feature = FeatureExtraction()

start_t = time.time()
X, y = feature.first_method(melanoma_train, ground_train)
feature_t = (time.time() - start_t)/60 # minutes

"""
------------------
"""

"""
Training Neural Network
-----------------------
"""

# Training the neural network with 83.3 % of the array features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.16666)

classifier = neural_network()
start_t = time.time()
classifier.fit(X_train, y_train)
classifier_t = (time.time() - start_t)/60 # minutes

score_test = classifier.score(X_test, y_test)
score_train = classifier.score(X_train, y_train)


"""
-----------------------
"""

"""
Classify test images
---------------
"""
melanoma_list = melanoma_test[0:1]
ground_list = ground_test[0:1]

seg, tim, dim = classify(melanoma_list, ground_list, feature, classifier, block=True)

"""
---------------
"""


"""
Accuracy
---------
"""
confmat = confusion_matrix(seg, ground_list)

local_err = local_error(confmat)
sensitivity, specificity, accuracy = total_error(local_err)

"""
---------
"""

"""
Measure of times of execution
-----------------------------
"""
tim = np.array(tim) # sec
dim = np.array(dim)
dim = dim[0:,0] * dim[0:,1]
t_by_pix = (tim*(10**6)) / dim # microsec / pix

tim /= 60 # min

total_time = (tim/60).sum() # total hours
mean_time = tim.mean() # mean minutes
std_time = tim.std() # std minutes


"""
-----------------------------
"""

"""
Saving values
-------------
"""
files = [f.split('.')[0]+'_classified.jpg' for f in melanoma_list]

path_save = 'resultados/red1/sin_preprocesar/test/'

for s, f in zip(seg, files):
    img = Image.fromarray(s)
    img.convert('L').save(path_save + f)

with open(path_save + 'Measures.txt', 'w') as output:
    output.write('---------------\n')
    output.write('---- RED 1 ----\n')
    output.write('---------------\n\n')
    output.write('Data Base: ' + cfg.melanoma_path + '\n')
    output.write('Number of images: ' + str(cfg.nImage) + '\n')
    output.write('Number of fields: ' + str(cfg.nCells) + '\n')
    output.write('Number of images to train: ' + str(len(melanoma_train)) + '\n')
    output.write('Number of image to test: ' + str(len(melanoma_test)) + '\n')
    output.write('Size of Train from Train_Images: ' + str(X_train.shape) + '\n')
    output.write('Size of Test from Train_Images: ' + str(X_test.shape) + '\n')
    output.write('Type of segmentation: block\n\n')
    output.write(classifier.__str__()+'\n\n')
    output.write('Final function value: ' + str(classifier.loss_)+'\n\n')
    output.write('-------------------------------------------------------------------------\n')
    output.write('Time of execution: \n')
    output.write('-------------------------------------------------------------------------\n\n')
    output.write('Feature Extraction: \n')
    output.write('\tTime: ' + str(feature_t) + ' min\n')
    output.write('Neural Network Training:\n')
    output.write('\tTime: ' + str(classifier_t) + ' min\n')
    output.write('Segmentation by image:\n')
    output.write('\tTotal: ' + str(total_time) + ' hrs\n')
    output.write('\tMean: ' + str(mean_time) + '+-' + str(std_time) + ' min\n')
    output.write('Segmentation by pixel:\n')
    output.write('\tMean: ' + str(t_by_pix.mean()) + '+-' + str(t_by_pix.std()) + ' mircosec/pix\n')
    output.write('-------------------------------------------------------------------------\n\n')
    output.write('Score:\n')
    output.write('\tX_train: ' + str(score_train) + '\n')
    output.write('\tX_test: ' + str(score_test) + '\n')
    output.write('-------------------------------------------------------------------------\n\n')
    output.write('Total error\n')
    output.write('\tSensitivity: ' + str(sensitivity[0]) + '+-' + str(sensitivity[1]) + '\n')
    output.write('\tSpecificity: ' + str(specificity[0]) + '+-' + str(specificity[1]) + '\n')
    output.write('\tAccuracy: ' + str(accuracy[0]) + '+-' + str(accuracy[1]) + '\n')
    output.write('-------------------------------------------------------------------------\n\n')
    output.write('Numero total de pixeles: ' + str(dim.sum()) + '\n')
    output.write('-------------------------------------------------------------------------\n\n')
    output.write('Local error: \n')
    output.write('\t[TP\tFP\tFN\tTN]|[sensitivity, specificity, accuracy]\t\n')
    for a, g, l, t, d in zip(confmat, ground_list, local_err, tim, dim):
        output.write(str(a) + '\t' + g + '\t' + str(l) + '\t' + str(t) + ' min' + '\t' + str(d) + ' pix\n')


"""
-------------
"""


"""
Classify train images
---------------------
"""
melanoma_list = melanoma_train[0:1]
ground_list = ground_train[0:1]

seg, tim, dim = classify(melanoma_list, ground_list, feature, classifier, block=True)

"""
---------------------
"""


"""
Accuracy
---------
"""
confmat = confusion_matrix(seg, ground_list)

local_err = local_error(confmat)
sensitivity, specificity, accuracy = total_error(local_err)

"""
---------
"""

"""
Measure of times of execution
-----------------------------
"""
tim = np.array(tim) # sec
dim = np.array(dim)
dim = dim[0:,0] * dim[0:,1]
t_by_pix = (tim*(10**6)) / dim # microsec / pix

tim /= 60 # min

total_time = (tim/60).sum() # total hours
mean_time = tim.mean() # mean minutes
std_time = tim.std() # std minutes


"""
-----------------------------
"""

"""
Saving values
-------------
"""
files = [f.split('.')[0]+'_classified.jpg' for f in melanoma_list]

path_save = 'resultados/red1/sin_preprocesar/train/'

for s, f in zip(seg, files):
    img = Image.fromarray(s)
    img.convert('L').save(path_save + f)

with open(path_save + 'Measures.txt', 'w') as output:
    output.write('---------------\n')
    output.write('---- RED 1 ----\n')
    output.write('---------------\n\n')
    output.write('Data Base: ' + cfg.melanoma_path + '\n')
    output.write('Number of images: ' + str(cfg.nImage) + '\n')
    output.write('Number of fields: ' + str(cfg.nCells) + '\n')
    output.write('Number of images to train: ' + str(len(melanoma_train)) + '\n')
    output.write('Number of image to test: ' + str(len(melanoma_test)) + '\n')
    output.write('Size of Train from Train_Images: ' + str(X_train.shape) + '\n')
    output.write('Size of Test from Train_Images: ' + str(X_test.shape) + '\n')
    output.write('Type of segmentation: block\n\n')
    output.write(classifier.__str__()+'\n\n')
    output.write('Final function value: ' + str(classifier.loss_)+'\n\n')
    output.write('-------------------------------------------------------------------------\n')
    output.write('Time of execution: \n')
    output.write('-------------------------------------------------------------------------\n\n')
    output.write('Feature Extraction: \n')
    output.write('\tTime: ' + str(feature_t) + ' min\n')
    output.write('Neural Network Training:\n')
    output.write('\tTime: ' + str(classifier_t) + ' min\n')
    output.write('Segmentation by image:\n')
    output.write('\tTotal: ' + str(total_time) + ' hrs\n')
    output.write('\tMean: ' + str(mean_time) + '+-' + str(std_time) + ' min\n')
    output.write('Segmentation by pixel:\n')
    output.write('\tMean: ' + str(t_by_pix.mean()) + '+-' + str(t_by_pix.std()) + ' mircosec/pix\n')
    output.write('-------------------------------------------------------------------------\n\n')
    output.write('Score:\n')
    output.write('\tX_train: ' + str(score_train) + '\n')
    output.write('\tX_test: ' + str(score_test) + '\n')
    output.write('-------------------------------------------------------------------------\n\n')
    output.write('Total error\n')
    output.write('\tSensitivity: ' + str(sensitivity[0]) + '+-' + str(sensitivity[1]) + '\n')
    output.write('\tSpecificity: ' + str(specificity[0]) + '+-' + str(specificity[1]) + '\n')
    output.write('\tAccuracy: ' + str(accuracy[0]) + '+-' + str(accuracy[1]) + '\n')
    output.write('-------------------------------------------------------------------------\n\n')
    output.write('Numero total de pixeles: ' + str(dim.sum()) + '\n')
    output.write('-------------------------------------------------------------------------\n\n')
    output.write('Local error: \n')
    output.write('\t[TP\tFP\tFN\tTN]|[sensitivity, specificity, accuracy]\t\n')
    for a, g, l, t, d in zip(confmat, ground_list, local_err, tim, dim):
        output.write(str(a) + '\t' + g + '\t' + str(l) + '\t' + str(t) + ' min' + '\t' + str(d) + ' pix\n')


"""
-------------
"""
