from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import config as cfg

"""
Implements diferent methods of learning which have a common interface to access
-------------------------------------------------------------------------------
"""


def neural_network():
    classifier = MLPClassifier()
    for (prop, value) in cfg.learningParams.items():
        setattr(classifier, prop, value)
    return classifier

