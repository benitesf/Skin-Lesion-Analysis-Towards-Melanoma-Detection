from sklearn.neural_network import MLPClassifier
import config as cfg


class NeuralNetwork:

    ## Instance a learning object, which is specified by a type parameter
    def __init__(self):
        self.clf = MLPClassifier()
        for (prop, value) in cfg.learningParams.items():  # Set all the user parameters
            setattr(self.clf, prop, value)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, T):
        return self.clf.predict(T)