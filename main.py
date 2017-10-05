# Import methods of features extraction
from features_extraction.feature_extraction import FeatureExtraction

# Import methods of learning
from learning.learning import neural_network

# Import methods of classification
from classification.classification import classify, confusion_matrix, estimate_error

#
from skimage import io
from PIL import Image

# Import util methods
from sklearn.model_selection import train_test_split
import util.dirhandler as dh
import config as cfg

"""
Get train and test set
----------------------
"""

all_melanoma = sorted(dh.get_file_name_dir(cfg.melanoma_path, cfg.melanoma_extension))
all_ground = sorted(dh.get_file_name_dir(cfg.ground_path, cfg.ground_extension))

melanoma_train, melanoma_test, ground_train, ground_test = train_test_split(all_melanoma, all_ground, test_size=0.30,
                                                                            random_state=15)

"""
----------------------
"""

"""
Feature Extraction
------------------
"""
feature = FeatureExtraction()

X_train, y_train = feature.first_method(melanoma_train, ground_train)

"""
------------------
"""

"""
Training Neural Network
-----------------------
"""

classifier = neural_network()
classifier.fit(X_train, y_train)

"""
-----------------------
"""


"""
Classify images
---------------
"""
melanoma_list = melanoma_test[0:10]
ground_list = ground_test[0:10]

seg = classify(melanoma_list, ground_list, feature, classifier)

confmat = confusion_matrix(seg, ground_list)
sensitivity, specificity, accuracy, roc = estimate_error(confmat)

files = [f.split('.')[0]+'_classified.jpg' for f in melanoma_list]

path_save = 'image/Classified/Segmentation_Preprocessed_Data/'

for s, f in zip(seg, files):
    img = Image.fromarray(s)
    img.convert('L').save(path_save + f)
    #io.imsave('image/Classified/'+f, img)

with open(path_save + 'lbfgs_first_method.txt', 'w') as output:
    output.write('lbfgs, first_method, block segmentation\n\n')
    output.write(classifier.__str__()+'\n\n')
    output.write(str(classifier.loss_)+'\n\n')
    output.write('VP\tFP\tFN\tVN\n')
    for a, g in zip(confmat, ground_list):
        output.write(str(a)+'\t'+g+'\n')
    output.write('\nSensitivity: ' + str(sensitivity))
    output.write('\nSpecificity: ' + str(specificity))
    output.write('\nAccuracy: ' + str(accuracy))

"""
---------------
"""
