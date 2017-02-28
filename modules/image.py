# Script Name		: image.py
# Author				: Benites Fernandez, Edson
# Created				: 27/02/17
# Last Modified	: 
# Version				: 1.0

# Modifications	: 1.1 - some modifications
#							  : 1.2 - some modifications

# Description		: Class to manage two images which are complementary, data and ground.
#									The purpose is help to calculate the blocks and its means, gabor filters, ...
#
#									Constructor recives the image paths, block dimension and the gabor kernels.
#									Also reads the two images and saves its float representations respectively,
#									block dimension, kernels and the image shape
#

import numpy as np
import random as rnd
from scipy import ndimage as ndi
from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray

class Image:	

	def __init__(self, path_data, path_ground, dim, kernels):	# Create object and initialize it
		try:
			self.image  = io.imread(path_data)
			self.ground = io.imread(path_ground)			
		except:
			print('Error. Can not read the image. Maybe wrong path.')
			return		
		self.image   = img_as_float(self.image)	
		self.ground  = img_as_float(self.ground)
		self.dim     = dim		
		self.kernels = kernels
		self.size    = self.image.shape	

	def img_block(self, pixel):						# Function to read a block dim x dim from the image																							
		fil_up     = pixel[0] - self.dim		#####
		fil_down   = pixel[0] + self.dim		# Calculate the block's indexs
		col_left   = pixel[1] - self.dim		#
		col_right  = pixel[1] + self.dim		#####		
		return self.image[fil_up : fil_down + 1, col_left : col_right + 1, :]	# Sum 1, because python's issues 

	def rnd_pix(self):		# Return a random central pixel of the block from the image 
		fil = [self.dim, self.size[0] - self.dim - 1]
		col = [self.dim, self.size[1] - self.dim - 1]
		return np.array([rnd.randint(fil[0], fil[1]), rnd.randint(col[0], col[1])])

	def mean_rgb(self, block):					# Function to calculte a mean rgb from a block of image			
		mean_r = np.mean(block[:,:,0])		# block[:,:,0].mean()						
		mean_g = np.mean(block[:,:,1])		# block[:,:,1].mean()	
		mean_b = np.mean(block[:,:,2])		# block[:,:,2].mean()
		return np.array([mean_r, mean_g, mean_b])
		
	def is_melanoma(self, pixel):		 # Return the pixel from ground (0 or 1)
		return self.ground[pixel[0], pixel[1]]

	def gabor(self, block):				# Applys the gabor kernels to the block
		img = rgb2gray(block) 	 
		return self.compute_feats(img)

	def compute_feats(self, img):		# Compute gabor kernels
		feats = np.zeros((len(self.kernels)), dtype = np.double)
		for k, kernel in enumerate(self.kernels):			
			filtered = ndi.convolve(img, kernel.real, mode = 'wrap')
			feats[k] = filtered.mean()
		return feats


