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

"""
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
"""
