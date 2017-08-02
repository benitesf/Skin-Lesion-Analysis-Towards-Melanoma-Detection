# Script Name			: config.py
# Author				: Benites Fernandez, Edson
# Created				: 27/02/17
# Last Modified			:
# Version				: 1.0

# Modifications			: 1.1 - some modifications
#						: 1.2 - some modifications

# Description			: Script to set up our start configuration like path of datasets
#

import util.verbose as v

# Activate the verbose, debug and warning function
logging = v.setUpVerbose()

#################################################
########## Initialize parameters ################
#################################################

# Train
train_data_path  = "image/train_data/"
data_ext   = "jpg"

train_ground_path = "image/train_ground/"
ground_ext  = "png"

# Test
test_data_path   = "image/test_data/"
test_ground_path = "image/test_ground/"


nImage    = 1000  # Number of images to work (1000)
nSample   = 40    # Number of samples per each image (40)
blockDim  = 12    # Block dimension is 25x25, so take ((dim-1)/2)
nCells    = 8     # Number of cells by each field of the training set

advanced_n_cells = 16

# Parameters to gabor kernels
frequency  = [0.6, 0.7]	# Spatial frequency of the harmonic function. Specified in pixels.
theta 		 = [0, 45]	    # Orientation in radians. If 0, the harmonic is in the x-direction.
#bandwidth = []
#sigma_x   = []
#sigma_y	 = []
#n_stds    = []
#offset    = []
gabor_params = [frequency, theta]
advanced_frequency = 0.5

## Learning Parameters
##
##
##  params = {'hidden_layer_size': , 'activation': , 'solver': , 'alpha': , 'batch_size': , 'learning_rate': ,\
##  'max_iter': ,'random_state': , 'shuffle': , 'tol': , 'learning_rate_init': , 'power_t': , 'verbose': ,\
##  'warm_start': , 'momentum': , 'nesterovs_momentum': , 'early_stopping': , 'validation_fraction': ,\
##  'beta_1': , 'beta_2': , 'epsilon': }
##
## For more information about the parameters see:
##  http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
learningParams = {'hidden_layer_size': (15, ), 'activation': 'logistic', 'solver': 'adam', 'alpha': 1e-5,
                  'max_iter': 300,'random_state': 1, 'shuffle': True, 'verbose': True}