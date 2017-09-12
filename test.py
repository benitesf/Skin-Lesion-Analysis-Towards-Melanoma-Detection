
from skimage import io
from matplotlib import pyplot as plt
from features_extraction import feature_extraction as FE
from learning import learning as LE

data0 = ["ISIC_0011227.jpg"]
data1 = ["ISIC_0011229.jpg"]

ground0 = ["ISIC_0011227_segmentation.png"]
ground1 = ["ISIC_0011229_segmentation.png"]

myFE = FE.get('MRG')
myLE = LE.get('NN')

X, y = myFE.getTestDataSet(data0, ground0)

"""TEST GABOR FILTER
from skimage.filters import gabor_kernel
import numpy as np

alfa = 45
theta = (alfa/360)*2*np.pi
gk = gabor_kernel(frequency=0.01, theta=theta)
plt.figure()
io.imshow(gk.real)
io.show()
"""

""""""

"""
plt.figure()
io.imshow()
io.show()
"""

