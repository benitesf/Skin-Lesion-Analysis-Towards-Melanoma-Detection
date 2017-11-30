"""
Paths and extension for images
------------------------------
"""

melanoma_path = 'image/ISIC-2017_Training_Data_Clean/'
melanoma_extension = 'jpg'

ground_path = 'image/ISIC-2017_Training_Part1_GroundTruth_Clean/'
ground_extension  = "png"


"""
Parameters for the data set
"""
nImage = 400  # Number of images to work (1000)
nSample = 40    # Number of samples per each image (40)
block = 25       # Block dimension is 25x25
nCells = 69     # Number of cells by each field of the training set

"""
Parameters for gabor kernels
"""
# First method
frequency = [0.4]
theta = [0, 45, 90, 135]
gabor_params = [frequency, theta]

# Second method
fmax = 1/2
v = 2
b = 1.177
ns = 2
nd = 4

"""
Parameters for the neural network
---------------------------------

solver lbfgs works better for small datasets.
solver adam works better for big datasets
"""
learningParams = {'hidden_layer_sizes': (136, ), 'activation': 'logistic', 'solver': 'lbfgs', 'alpha': 1e-5,
                  'max_iter': 300,'random_state': 1, 'shuffle': True, 'verbose': True}

##  params = {'hidden_layer_size': , 'activation': , 'solver': , 'alpha': , 'batch_size': , 'learning_rate': ,\
##  'max_iter': ,'random_state': , 'shuffle': , 'tol': , 'learning_rate_init': , 'power_t': , 'verbose': ,\
##  'warm_start': , 'momentum': , 'nesterovs_momentum': , 'early_stopping': , 'validation_fraction': ,\
##  'beta_1': , 'beta_2': , 'epsilon': }
##
## For more information about the parameters see:
##  http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
