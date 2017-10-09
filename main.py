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
import sys

"""
Get train and test set
----------------------
"""

all_melanoma = sorted(dh.get_file_name_dir(cfg.melanoma_path, cfg.melanoma_extension))
all_ground = sorted(dh.get_file_name_dir(cfg.ground_path, cfg.ground_extension))

melanoma_train, melanoma_test, ground_train, ground_test = train_test_split(all_melanoma, all_ground, test_size=0.25,
                                                                            random_state=15)


"""
----------------------
"""

"""
Feature Extraction
------------------
"""
feature = FeatureExtraction()

X, y = feature.first_method(melanoma_train, ground_train)

"""
------------------
"""

"""
Training Neural Network
-----------------------
"""

# Training the neural network with 83.3 % of the array features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.166)

classifier = neural_network()
classifier.fit(X_train, y_train)

score_test = classifier.score(X_test, y_test)
score_train = classifier.score(X_train, y_train)


"""
-----------------------
"""

"""
Classify images
---------------
"""
melanoma_list = melanoma_test
ground_list = ground_test

seg, tim = classify(melanoma_list, ground_list, feature, classifier)

"""
---------------
"""

"""
Accuracys
---------
"""
confmat = confusion_matrix(seg, ground_list)

local_err = local_error(confmat)
sensitivity, specificity, accuracy = total_error(local_err)

tim = np.array(tim)/120
total_time = tim.sum()
mean_time = tim.mean()
std_time = tim.std()

"""
---------
"""

"""
Saving values
-------------
"""
files = [f.split('.')[0]+'_classified.jpg' for f in melanoma_list]

path_save = 'image/First_Test/Normal_Data/'

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
    output.write('Size of Train from Train_Images : ' + str(X_train.shape) + '\n')
    output.write('Size of Test from Train_Images:' + str(X_test.shape) + '\n')
    output.write('Type of segmentation: block\n\n')
    output.write(classifier.__str__()+'\n\n')
    output.write('Final function value :' + str(classifier.loss_)+'\n\n')
    output.write('-------------------------------------------------------------------------\n')
    output.write('Tiempos de ejecución:\n')
    output.write('\tTotal: ' + str(total_time) + '\n')
    output.write('\tMean: ' + str(mean_time) + '+-' + str(std_time) + '\n')
    output.write('-------------------------------------------------------------------------\n\n')
    output.write('Score:\n')
    output.write('\tX_train: ' + str(score_train) + '\n')
    output.write('\tX_test: ' + str(score_test) + '\n')
    output.write('-------------------------------------------------------------------------\n')
    output.write('Total error\n')
    output.write('\tSensitivity: ' + str(sensitivity[0]) + '+-' + str(sensitivity[1]) + '\n')
    output.write('\tSpecificity: ' + str(specificity[0]) + '+-' + str(specificity[1]) + '\n')
    output.write('\tAccuracy: ' + str(accuracy[0]) + '+-' + str(accuracy[1]) + '\n')
    output.write('-------------------------------------------------------------------------\n\n')
    output.write('\tTP\tFP\tFN\tTN\n')
    for a, g, l in zip(confmat, ground_list, local_err):
        output.write(str(a) + '\t' + g + '\t' + str(l) + '\n')


"""
-------------
"""
