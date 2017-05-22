from learning.NeuralNetwork import NeuralNetwork


# Instance a specific learning object
def get(type):
    featureType = {'NN': neuralNetwork
                   }
    return featureType[type]()


# Instance a neural network object
def neuralNetwork():
    return NeuralNetwork()

###########
# Here we can add more functions to instace any learning class
##########

