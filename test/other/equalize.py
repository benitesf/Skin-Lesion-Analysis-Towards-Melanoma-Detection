from skimage import io, exposure, img_as_ubyte
from skimage.color import rgb2grey
import matplotlib.pyplot as plt
import sys, os

sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection")

path = 'image/image_for_test/'

melanoma = ('ISIC_0000047.jpg',
            'ISIC_0000049.jpg',
            'ISIC_0000055.jpg',
            'ISIC_0000066.jpg',
            'ISIC_0000077.jpg')

image = io.imread(path + melanoma[1])

image = rgb2grey(image)
image = img_as_ubyte(image)

hist, bin_centers = exposure.histogram(image)

plt.hist(hist, bin_centers)
plt.xlabel('Pixel intensity')
plt.ylabel('Number of pixels')
plt.show()



