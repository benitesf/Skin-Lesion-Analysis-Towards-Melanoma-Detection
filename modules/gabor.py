# Script Name		: gabor.py
# Author				: Benites Fernandez, Edson
# Created				: 28/02/17
# Last Modified	: 
# Version				: 1.0

# Modifications	: 1.1 - some modifications
#							  : 1.2 - some modifications

# Description		: Create gabor filters based on the parameters
#

import numpy as np
from skimage.filters import gabor_kernel

def kernels(params):
	kernels = []		
	for frequency in params[0]:
		for theta in params[1]:
			theta = (theta / 360.) * 2. * np.pi
			kernel = gabor_kernel(frequency, theta = theta, bandwidth = 5)
			kernels.append(kernel) 
	return kernels