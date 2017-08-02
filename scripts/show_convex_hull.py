import sys, os
sys.path.append("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection/")
from features_extraction.majorAxis import get_theta_convex_hull
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc


os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection")
ground = 'ISIC_0000000_segmentation.png'

image = scipy.misc.imread('image/train_ground/ISIC_0000000_segmentation.png', flatten=True)
theta, hull = get_theta_convex_hull(ground)

y, x = np.nonzero(image)

points = np.array([x, y]).transpose()

plt.plot(x, y, 'k.')

for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.plot(points[simplex, 0], points[simplex, 1], 'r-')

plt.axis('equal')
plt.gca().invert_yaxis()
plt.show()
