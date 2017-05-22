import numpy as np
import config as cfg
from classes.image import Image
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
import time

class MeanRgbGaborExtraction:

    def __init__(self):
        pass

    # This methos obtains the data set from the train list
    def getTrainDataSet(self, data_list, ground_list):
        return self.getDataSet(data_list, ground_list, cfg.train_data_path, cfg.train_ground_path)

    # This methos obtains the data set from the test list
    def getTestDataSet(self, data_list, ground_list):
        return self.getDataSet(data_list, ground_list, cfg.test_data_path, cfg.test_ground_path)

    # This method return a data set array (nSample*nImage, nCells)
    def getDataSet(self, data, ground, data_path, ground_path):

        self.gaborKernels = self.kernels(cfg.gabParams)

        X = np.zeros((cfg.nSample * cfg.nImage, cfg.nCells - 1))
        y = np.zeros((cfg.nSample * cfg.nImage,))

        for (data_name, ground_name, image_index) in zip(data, ground, range(cfg.nImage)):
            curr_img = Image(data_path + data_name, ground_path + ground_name, cfg.blockDim)
            for sample_index in range(cfg.nSample):
                index = image_index * cfg.nSample + sample_index
                X[index, :], y[index] = self.features(curr_img)

        return X, y.astype(int)

    # This method obtains the features of a image
    def features(self, currImg):
        pix = currImg.randomCentralPixel(border=True)  # [fil, col] get a random pixel from the current image
        blk = currImg.getBlock(pix)  # 25x25 get the block 25x25 from the central pixel
        rgb = self.meanRgb(blk)  # [mean r, mean g, mean b] calculate means
        #start_time = time.time()
        gab = self.gaborFilter(blk)  # [gab 0, gab 1, gab 2, gab 3] calculate mean convolve gabor filters
        #print("--- %s seconds ---" % (time.time() - start_time))
        mel = currImg.getGroundPixel(pix)  # [0 or 1] get 0 if current pixel is not melanoma, and 1 otherwise
        return [*rgb, *gab], mel

    # Calculates the means of a image RGB, return 3 values
    def meanRgb(self, block):  # Function to calculte a mean rgb from a block of image
        meanR = np.mean(block[:, :, 0])  # block[:,:,0].mean()
        meanG = np.mean(block[:, :, 1])  # block[:,:,1].mean()
        meanB = np.mean(block[:, :, 2])  # block[:,:,2].mean()
        return [meanR, meanG, meanB]

    # Applys the gabor kernels to a block of pixels
    def gaborFilter(self, block):
        img = rgb2gray(block)
        return self.computeFeats(img)

    def computeFeats(self, img):  # Compute gabor kernels
        feats = np.zeros((len(self.gaborKernels)), dtype=np.double)
        for k, kernel in enumerate(self.gaborKernels):
            filtered = ndi.convolve(img, kernel.real, mode='wrap')
            feats[k] = filtered.mean()
        return feats

    # This method calculate the gabor kernels and return it
    def kernels(self,params):
        kernels = []
        for frequency in params[0]:
            for theta in params[1]:
                theta = (theta / 360.) * 2. * np.pi
                kernel = gabor_kernel(frequency, theta=theta, bandwidth=5)
                kernels.append(kernel)
        return kernels