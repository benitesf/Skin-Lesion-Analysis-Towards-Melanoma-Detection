from classes.image import Image
from classes.image import reader_img
from skimage.measure import compare_mse
import numpy as np
import config as cfg

class Classification:

    # This constructor saves a learning object
    def __init__(self, learning, featureExtraction):
        self.learning          = learning
        self.featureExtraction = featureExtraction

    # Set the learning method
    def setLearnig(self, learning):
        self.learning = learning

    # Set the feature extraction method
    def setFeatureExtraction(self, featureExtraction):
        self.featureExtraction = featureExtraction

    # Classifys and calculates the mse between the image classified and the ground image
    def accurate(self, data_set, ground_set, set):
        acc = []

        if (set == 'test'):
            path = cfg.test_ground_path
        elif (set == 'train'):
            path = cfg.tra_ground_path

        for (data, ground) in zip (data_set, ground_set):
            c = self.classify_test(data, block=True)
            g = reader_img(path + ground)
            acc.append(compare_mse(c,g))
        return acc

    # Classify the train set
    def classify_train(self, data, block=None):
        if (block is None) or (block) :
            return self.classify_per_block(data, cfg.train_data_path)
        else:
            return self.classify_per_pixel(data, cfg.train_data_path)

    # Classify the test set
    def classify_test(self, data, block=None):
        if (block is None) or (block):
            return self.classify_per_block(data, cfg.test_data_path)
        else:
            return self.classify_per_pixel(data, cfg.test_data_path)

    # Classify a image per pixels
    def classify_per_pixel(self, data, path):
        image = Image(path + data, 'None', cfg.blockDim)
        originalSize = image.getOriginalSize()

        imageClassify = np.zeros((originalSize[0], originalSize[1]))

        startRow = cfg.blockDim
        endRow = image.image.shape[0] - cfg.blockDim
        startCol = cfg.blockDim
        endCol = image.image.shape[1] - cfg.blockDim

        for row in range(startRow, endRow):
            for col in range(startCol, endCol):
                blk = image.getBlock([row, col])  # 25x25 get the block 25x25 from the central pixel
                rgb = self.featureExtraction.meanRgb(blk)  # [mean r, mean g, mean b] calculate means
                gab = self.featureExtraction.gaborFilter(blk)  # [gab 0, gab 1, gab 2, gab 3] calculate mean convolve gabor filters
                set = [*rgb, *gab]
                imageClassify[row - cfg.blockDim, col - cfg.blockDim] = self.learning.predict([set])
        return imageClassify

    # Classify a image per blocks
    def classify_per_block(self, data, path):
        image = Image(path + data, 'None', cfg.blockDim)
        originalSize = image.getOriginalSize()

        imageClassify = np.zeros((originalSize[0], originalSize[1]))

        ratio = 4 # block dimension is (9x9)

        startRow = cfg.blockDim + ratio
        endRow = image.image.shape[0] - cfg.blockDim - ratio
        startCol = cfg.blockDim + ratio
        endCol = image.image.shape[1] - cfg.blockDim - ratio

        for row in range(startRow, endRow, ratio + 1):
            for col in range(startCol, endCol, ratio + 1):
                blk = image.getBlock([row, col])  # 25x25 get the block 25x25 from the central pixel
                rgb = self.featureExtraction.meanRgb(blk)  # [mean r, mean g, mean b] calculate means
                gab = self.featureExtraction.gaborFilter(blk)  # [gab 0, gab 1, gab 2, gab 3] calculate mean convolve gabor filters
                set = [*rgb, *gab]
                pred = self.learning.predict([set])
                imageClassify[row - cfg.blockDim : row - cfg.blockDim + (ratio*2)+1 , col - cfg.blockDim : col - cfg.blockDim + (ratio*2) + 1] = pred
        return imageClassify

    # Calculates the mean-square error between two images
    def mean_squared_error(self, im1, im2):
        from skimage.measure import compare_mse
        return compare_mse(im1, im2)