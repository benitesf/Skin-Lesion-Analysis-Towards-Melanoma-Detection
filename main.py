# Import methods of features extraction
from features_extraction.feature_extraction import FeatureExtraction

# Import methods of learning
from learning.learning import neural_network

# Import methods of classification
from classification.classification import classify

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

melanoma_train, melanoma_test, ground_train, ground_test = train_test_split(all_melanoma, all_ground, test_size=0.33,
                                                                            random_state=42)

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
melanoma_list = [melanoma_test[0]]
ground_list = [ground_test[0]]

seg = classify(melanoma_list, ground_list, feature, classifier)

"""
---------------
"""
