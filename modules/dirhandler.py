# Script Name		: dirhandler.py
# Author				: Benites Fernandez, Edson
# Created				: 27/02/17
# Last Modified	: 
# Version				: 1.0

# Modifications	: 1.1 - some modifications
#							  : 1.2 - some modifications

# Description		: Implement functios to manage directories.
#									Create, remove, move, rename, ... 
#									Also allows to get the number and names of files in a directory 
#									with a specific extension.
#

import os, glob

# Create a new directory
def make_dir(path):
	os.system("mkdir "+path)	

# Change directory
def change_dir(path):
	os.system("cd "+path)

# Remove a dir
def remove_dir(path):
	os.system("rmdir "+path)	

# Remove a dir (recursive and force)
def removeRf_dir(path):	
	os.system("rm "+path+" -Rf")	

# Move or rename a dir
def move_dir(src, dest):
	os.system("mv "+src+" "+dest)

# Copy file
def copy_file(src, dest):
	os.system("cp "+src+" "+dest)

# Print dir names and its contents files
def print_list_dir(rootDir):
	for dirName, subDirList, fileList in os.walk(rootDir):
		print('Directorio encontrado: %s' % dirName)
		for fname in fileList:
			print('\t%s' % fname)

# Count the number of files with certain extension
def num_files_dir(rootDir, ext):
	return len(glob.glob1(rootDir,"*."+ext))

# Return file names with certain extension
def get_file_name_dir(rootDir, ext):
	return glob.glob1(rootDir,"*."+ext)	