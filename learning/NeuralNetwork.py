from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import config as cfg


class NeuralNetwork:

    ## Instance a learning object, which is specified by a type parameter
    def __init__(self):
        self.clf = MLPClassifier()
        for (prop, value) in cfg.learningParams.items():  # Set all the user parameters
            setattr(self.clf, prop, value)

    # This method use cross validation to get the best fitness
    def best_validation(self, X, y, cv=5):
        return cross_val_score(self.clf, X, y)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def score(self, X, y):
        return self.clf.score(X, y)

    def predict(self, T):
        return self.clf.predict(T)

    def loss(self):
        return self.clf.loss_
