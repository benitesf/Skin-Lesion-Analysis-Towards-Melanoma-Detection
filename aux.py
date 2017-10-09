from util import dirhandler as dh
import shutil, os


"""
Script para leer el nombre de las imágenes de la carpeta de datos y pasar sus correspondientes
ground_truth a otra carpeta
"""

path_file = 'image/ISIC-2017_Training_Data_Wierd'

images = dh.get_file_name_dir(path_file, 'jpg')
images = [i.split('.')[0]+'_segmentation.png'  for i in images]

path_src = 'image/ISIC-2017_Training_Part1_GroundTruth'
path_dst = 'image/ISIC-2017_Training_Part1_GroundTruth_Wierd'

for i in images:
    shutil.move(os.path.join(path_src,i), path_dst)