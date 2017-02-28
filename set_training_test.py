# Script Name		: set_training_test.py
# Author				: Benites Fernandez, Edson
# Created				: 27/02/17
# Last Modified	: 
# Version				: 1.0

# Modifications	: 1.1 - some modifications
#							  : 1.2 - some modifications

# Description		: Script to create de necesary directories
#

#	WARNING				: THIS SCRIPT HAVE A SPECIFIC PARAMETERS TO CREATE DIRECTORIES ON MY MACHINE
#									MAY YOU NEED TO CHANGE IT, IF YOU WANT TO USE IT
#

import modules.dirhandler as dh

if __name__ == '__main__':
	# Create train_data and test_data directory
	dh.make_dir("image/train_data")
	dh.make_dir("image/test_data")

	# Create train_ground and test_ground directory
	dh.make_dir("image/train_ground")
	dh.make_dir("image/test_ground")	

	# Get jpg and png file names
	training_files = sorted(dh.get_file_name_dir("image/ISIC-2017_Training_Data", "jpg"))
	ground_files = sorted(dh.get_file_name_dir("image/ISIC-2017_Training_Part1_GroundTruth", "png"))	

	# Copy 1000 files from training to train and test
	# Here you can copy or move the files (move_dir(src, dest))
	for name in training_files[0:1000]:		
		dh.copy_file("image/ISIC-2017_Training_Data/"+name, "image/train_data")	

	for name in training_files[1000:]:		
		dh.copy_file("image/ISIC-2017_Training_Data/"+name, "image/test_data")

	# Copy 1000 files from ground to train and test
	# Here you can copy or move the files (move_dir(src, dest))
	for name in ground_files[0:1000]:		
		dh.copy_file("image/ISIC-2017_Training_Part1_GroundTruth/"+name, "image/train_ground")	

	for name in ground_files[1000:]:		
		dh.copy_file("image/ISIC-2017_Training_Part1_GroundTruth/"+name, "image/test_ground")	

	
	

