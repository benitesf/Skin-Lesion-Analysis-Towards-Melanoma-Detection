import numpy as np
import config as cfg
from features_extraction.majorAxis import get_theta
from util.image import Image
from skimage.color import rgb2gray, rgb2hsv
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel


class AdvancedExtraction:

    def __init__(self):
        pass

# This methos obtains the data set from a list
    def get_data_set(self, data_list, ground_list=None, type=None):
        data_path = None
        ground_path = None
        if type is not None:
            if type == "train":
                data_path = cfg.train_data_path
                ground_path = cfg.train_ground_path
            elif type == "test":
                data_path = cfg.test_data_path
                ground_path = cfg.test_ground_path
            return self.build_array(data_list, ground_list, data_path, ground_path)
        else:
            print('Expected a type value. Posible values: train or test')
            raise AttributeError

    # This method return a data set array (nSample*nImage, nCells)
    def build_array(self, data, ground, data_path, ground_path):

        X = np.zeros((cfg.nSample * cfg.nImage, cfg.advanced_n_cells))
        y = np.zeros((cfg.nSample * cfg.nImage,))

        for (data_name, ground_name, image_index) in zip(data, ground, range(cfg.nImage)):
            img = Image(data_path + data_name, ground_path + ground_name, cfg.blockDim)
            self.set_theta(ground_path + ground_name, True)
            self.set_kernel()
            for sample_index in range(cfg.nSample):
                index = image_index * cfg.nSample + sample_index
                feats = self.features(img)
                X[index, 0:15] = feats[0]
                X[index, 15] = feats[1]
                y[index] = feats[2]
        return X, y.astype(int)

    # This method obtains the features of a image
    def features(self, img):
        pix = img.random_central_pixel(border=True)  # [fil, col] get a random pixel from the current image
        blk = img.get_block(pix)  # 25x25 get the block 25x25 from the central pixel
        mm  = self.max_min(blk)
        hsv = self.mean_hsv(blk)
        rgb = self.mean_rgb(blk)  # [mean r, mean g, mean b] calculate means
        std = self.standar_deviation(blk)
        gab = self.gabor_filter(blk)
        mel = img.get_ground_pixel(pix)  # [0 or 1] get 0 if current pixel is not melanoma, and 1 otherwise
        return [*mm, *hsv, *rgb, *std], gab, mel

    def max_min(self, block):
        minR = np.min(block[:, :, 0])
        minG = np.min(block[:, :, 1])
        minB = np.min(block[:, :, 2])

        maxR = np.max(block[:, :, 0])
        maxG = np.max(block[:, :, 1])
        maxB = np.max(block[:, :, 2])

        return [maxR, maxG, maxB, minR, minG, minB]

    def mean_hsv(self, block):
        img = rgb2hsv(block)
        meanH = np.mean(img[:, :, 0])
        meanS = np.mean(img[:, :, 1])
        meanV = np.mean(img[:, :, 2])
        return [meanH, meanS, meanV]

    def standar_deviation(self, block):
        stdR = np.std(block[:, :, 0])
        stdG = np.std(block[:, :, 1])
        stdB = np.std(block[:, :, 2])
        return [stdR, stdG, stdB]

    # Calculates the means of a image RGB, return 3 values
    def mean_rgb(self, block):  # Function to calculte a mean rgb from a block of image
        meanR = np.mean(block[:, :, 0])  # block[:,:,0].mean()
        meanG = np.mean(block[:, :, 1])  # block[:,:,1].mean()
        meanB = np.mean(block[:, :, 2])  # block[:,:,2].mean()
        return [meanR, meanG, meanB]

    # Applys the gabor kernels to a block of pixels
    def gabor_filter(self, block):
        img = rgb2gray(block)
        return self.compute_feats(img)

    def set_theta(self, path_image, bol):
        self.theta = get_theta(path_image, bol)

    # This method calculate the gabor kernel and return it
    def set_kernel(self):
        self.kernel = gabor_kernel(cfg.advanced_frequency, theta=self.theta, bandwidth=5)

    def compute_feats(self, img):  # Compute gabor kernels
        filtered = ndi.convolve(img, self.kernel.real, mode='wrap')
        return filtered.mean()