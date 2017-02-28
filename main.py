# Script Name		: main.py
# Author				: Benites Fernandez, Edson
# Created				: 27/02/17
# Last Modified	: 
# Version				: 1.0

# Modifications	: 1.1 - some modifications
#							  : 1.2 - some modifications

# Description		: Main script to run the system
#

from modules.image import Image  	
import modules.verbose as v     
import modules.dirhandler as dh
import modules.gabor as gab
import random as rnd
import numpy as np


#activate the verbose, debug and warning function
logging = v.setUpVerbose()

#################################################
########## Initialize parameters ################
#################################################
data_path  = "image/train_data/"
data_ext   = "jpg"

ground_path = "image/train_ground/"
ground_ext  = "png"

n_image    = 1000  # Number of images to work
n_sample   = 40    # Number of samples per each image									
block_dim  = 12    # Block dimension is 25x25, so take (dim-1/2)

# parameters to gabor kernels
frequency  = [0.05, 0.25]	# Spatial frequency of the harmonic function. Specified in pixels. 
theta 		 = [0, 45]			# Orientation in radians. If 0, the harmonic is in the x-direction.
#bandwidth = []						 
#sigma_x   = []						 
#sigma_y	 = []						
#n_stds    = []						
#offset    = []						
gab_params = [frequency, theta]
#################################################

# Get all the image names from train_data and train_ground directory
file_name_train_data   = sorted(dh.get_file_name_dir(data_path, data_ext))
file_name_train_ground = sorted(dh.get_file_name_dir(ground_path, ground_ext))

# Create train set to store [mean r, mean g, mean b, gabor 0, gabor 1, gabor 2, gabor 3, y]
train_set = np.zeros((n_sample * n_image, 8))

# Create gabor kernels with the specific parameters 
kernels  = gab.kernels(gab_params)

##################################################
######### Create the train set ###################
##################################################
for data_name, ground_name, image_index in zip(file_name_train_data, file_name_train_ground, range(n_image)):
	
	crnt_img = Image(data_path + data_name, ground_path + ground_name, block_dim, kernels)

	for sample_index in range(n_sample): 		

		pix = crnt_img.rnd_pix()					# [fil, col] get a random pixel from the current image				
		blk = crnt_img.img_block(pix)			# 25x25 get the block 25x25 from the central pixel
		rgb = crnt_img.mean_rgb(blk)			# [mean r, mean g, mean b] calculate means
		gab = crnt_img.gabor(blk)					# [gab 0, gab 1, gab 2, gab 3] calculate mean convolve gabor filters
		mel = crnt_img.is_melanoma(pix)		# [0 or 1] get 0 if current pixel is not melanoma, and 1 otherwise
		
		train_set[image_index * n_sample + sample_index, 0:3] = rgb[:]  ###
		train_set[image_index * n_sample + sample_index, 3:7] = gab[:]	#save the 8 values
		train_set[image_index * n_sample + sample_index, 7]   = mel 		### 

##################################################
############## Neural Network ####################
##################################################


##################################################
######### Training Neural Network ################
##################################################


##################################################
############## Classify images ###################
##################################################
