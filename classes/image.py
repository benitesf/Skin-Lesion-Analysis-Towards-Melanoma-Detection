# Script Name		: image.py
# Author				: Benites Fernandez, Edson
# Created				: 27/02/17
# Last Modified	: 
# Version				: 1.0

# Modifications	: 1.1 - some modifications
#							  : 1.2 - some modifications

# Description		: Class to manage two images which are complementary, data and ground.
#					  The purpose is help to calculate the blocks and its means, gabor filters, ...
#
#									Constructor recives the image paths, block dimension and the gabor kernels.
#									Also reads the two images and saves its float representations respectively,
#									block dimension, kernels and the image shape
#

import numpy as np
import random as rnd
from skimage import io
from skimage.util import img_as_float
from skimage.transform import resize

class Image:	

	def __init__(self, dataPath, groundPath, blockDim):	# Create object and initialize it
		try:
			self.image    = img_as_float(io.imread(dataPath))
			self.blockDim = blockDim
			self.image    = self.extendImage(self.image)
			try:
				self.ground = img_as_float(io.imread(groundPath))
				self.ground = self.extendImage(self.ground)
			except:
				print('Warning. Can not read the ground image. Maybe wrong path.')
		except:
			print('Error. Can not read the image. Maybe wrong path.')
			return

	##
	# This method extend a image, create a new borders
	##
	def extendImage(self, image):
		extention = (self.blockDim * 2) + 1
		return resize(image, (image.shape[0] + extention, image.shape[1] + extention), mode='symmetric')

	##
	# This function return a block of pixels from a certain center pixel
	# the method first of all checks if the block gets out of bounds
	# if the block is out side of the threshold then replicate the missing pixels like a mirror
	##
	def getBlock(self, pixel):						# Function to read a block dim x dim from the image
		topRow      = pixel[0] - self.blockDim		#####
		bottomRow   = pixel[0] + self.blockDim		# Calculate the block's indexs
		leftCol     = pixel[1] - self.blockDim		#
		rightCol    = pixel[1] + self.blockDim		#####

		return self.image[topRow: bottomRow + 1, leftCol: rightCol + 1, :]  # Sum 1, because python's issues

	##
	# This function return a random pixel from the image
	# It uses a boolean border parameter to determine the thresholds of the image where it go to take the random pixel
	# If border is True, then the threshold pixel go from [0,0] to [sizeImage - 1, sizeImage - 1]
	# If border is False, then the method just return a pixel such that when we ask for a block we dont take out of bound pixels
	##
	def randomCentralPixel(self, border):		# Return a random central pixel of the block from the image
		row = [self.blockDim, self.image.shape[0] - self.blockDim - 1]
		col = [self.blockDim, self.image.shape[1] - self.blockDim - 1]
		if not border:
			row = [self.blockDim * 2, self.image.shape[0] - (self.blockDim * 2) - 1]
			col = [self.blockDim * 2, self.image.shape[1] - (self.blockDim * 2) - 1]
		return np.array([rnd.randint(row[0], row[1]), rnd.randint(col[0], col[1])])


	##
	# Return a certain pixel from the ground image
	##
	def getGroundPixel(self, pixel):		 # Return the pixel from ground (0 or 1)
		return self.ground[pixel[0], pixel[1]]

	##
	# Return a certain pixel from the data image
	##
	def getImagePixel(self, pixel):			# Return the pixel from image
		return self.image[pixel[0], pixel[1]]

	##
	# Return the size of the original image
	##
	def getOriginalSize(self):
		return [self.image.shape[0] - (self.blockDim * 2), self.image.shape[1] - (self.blockDim * 2)]


def reader_img(path):
	return io.imread(path)