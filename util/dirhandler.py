import os, glob

"""
Implements util methods to manage directorys and files
"""


def make_dir(path):
    """
	Create a new directory
	"""
    os.system("mkdir " + path)


def change_dir(path):
    """
	Changes to a another directory path
	"""
    os.system("cd " + path)


def remove_dir(path):
    """
	Remove a certain directory
	"""
    os.system("rmdir " + path)


def removeRf_dir(path):
    """
	Remove a dir (recursive and force)
	"""
    os.system("rm " + path + " -Rf")


def move_dir(src, dest):
    """
	Move or rename a dir
	"""
    os.system("mv " + src + " " + dest)


def copy_file(src, dest):
    """
	Copy a file	to a certain directory
	"""
    os.system("cp " + src + " " + dest)


def print_list_dir(rootDir):
    """
	Print dir names and its contents files
	"""
    for dirName, subDirList, fileList in os.walk(rootDir):
        print('Directorio encontrado: %s' % dirName)
        for fname in fileList:
            print('\t%s' % fname)


def num_files_dir(rootDir, ext):
    """
	Count the number of files with certain extension
	"""
    return len(glob.glob1(rootDir, "*." + ext))


def get_file_name_dir(rootDir, ext):
    """
	Return file names with certain extension
	"""
    return glob.glob1(rootDir, "*." + ext)
