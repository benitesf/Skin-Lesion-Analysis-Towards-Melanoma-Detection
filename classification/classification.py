from util.image import Image
from util.image import read_image
from skimage.measure import compare_mse
from skimage import io
from features_extraction.threshold_otsu import get_segmentation
import numpy as np
import config as cfg
import os


class Classification:
    # This constructor saves a learning object and feature extraction method
    def __init__(self, learning, feature):
        self.learning = learning
        self.feature = feature

    # Set the learning method
    def setLearnig(self, learning):
        self.learning = learning

    # Set the feature extraction method
    def setFeatureExtraction(self, feature):
        self.feature = feature

    # Calculate the accuracy and save the segmentation image
    def accurate_and_segmentation(self, data_set, ground_set, set=None, string=None):
        size = len(data_set)
        acc = np.zeros((size, 4), dtype=int)

        data_path = None
        ground_path = None

        if set is not None:
            if set == 'test':
                data_path   = cfg.test_data_path
                ground_path = cfg.test_ground_path
            elif set == 'train':
                data_path   = cfg.train_data_path
                ground_path = cfg.train_ground_path

            cont = 0

            for (data, ground) in zip(data_set, ground_set):
                c = self.classify_advanced(data_path + data, block=True)
                g = read_image(ground_path + ground)
                self.save_segmentation(data, c, string)
                acc[cont, :] = self.compare_ground_truth(g, c)
                cont += 1
            return acc
        else:
            print('Expected a set value. Posible values: train or test')
            raise AttributeError

    # Classify and calculates the mse between the image classified and the ground image
    def accurate(self, data_set, ground_set, set=None):
        acc = np.zeros((15, 4), dtype=int)
        data_path = None
        ground_path = None
        if set is not None:
            if set == 'test':
                data_path = cfg.test_data_path
                ground_path = cfg.test_ground_path
            elif set == 'train':
                data_path = cfg.train_data_path
                ground_path = cfg.train_ground_path
            cont = 0
            for (data, ground) in zip(data_set, ground_set):
                c = self.classify_RGB(data_path + data, block=False)
                g = read_image(ground_path + ground)
                #self.save_accurate(data, c)
                self.save_segmentation(data, c, '')
                acc[cont, :] = self.compare_ground_truth(g, c)
                cont += 1
            return acc
        else:
            print('Expected a set value. Posible values: train or test')
            raise AttributeError


    # Classify a data set
    def classify_RGB(self, data, block=None):
        image = Image(data, 'None', cfg.blockDim)
        original_size = image.get_original_size()

        image_classified = np.zeros((original_size[0], original_size[1]))

        start_row = cfg.blockDim
        end_row = image.image.shape[0] - cfg.blockDim
        start_col = cfg.blockDim
        end_col = image.image.shape[1] - cfg.blockDim

        if (block is None) or (block):
            ratio = 4  # block dimension is (9x9)

            start_row += ratio
            end_row -= ratio
            start_col += ratio
            end_col -= ratio

            for row in range(start_row, end_row, ratio + 1):
                for col in range(start_col, end_col, ratio + 1):
                    blk = image.get_block([row, col])  # 25x25 get the block 25x25 from the central pixel
                    rgb = self.feature.mean_rgb(blk)  # [mean r, mean g, mean b] calculate means
                    gab = self.feature.gabor_filter(blk)  # [gab 0, gab 1, gab 2, gab 3] calculate mean convolve gabor filters
                    set = [*rgb, *gab]
                    pred = self.learning.predict([set])
                    image_classified[row - cfg.blockDim: row - cfg.blockDim + (ratio * 2) + 1,
                    col - cfg.blockDim: col - cfg.blockDim + (ratio * 2) + 1] = pred
            return image_classified
        else:
            print("Classifying pixel per pixel")
            for row in range(start_row, end_row):
                for col in range(start_col, end_col):
                    blk = image.get_block([row, col]) # 25x25 get the block 25x25 from the central pixel
                    rgb = self.feature.mean_rgb(blk)  # [mean r, mean g, mean b] calculate means
                    gab = self.feature.gabor_filter(blk) # [gab 0, gab 1, gab 2, gab 3] calculate mean convolve gabor filters
                    set = [*rgb, *gab]
                    image_classified[row - cfg.blockDim, col - cfg.blockDim] = self.learning.predict([set])
            return image_classified

    def classify_advanced(self, data, block=None):
        image = Image(data, 'None', cfg.blockDim)
        ground = get_segmentation(data)

        self.feature.set_theta(ground, False)
        self.feature.set_kernel()

        original_size = image.get_original_size()

        image_classified = np.zeros((original_size[0], original_size[1]))

        start_row = cfg.blockDim
        end_row = image.image.shape[0] - cfg.blockDim
        start_col = cfg.blockDim
        end_col = image.image.shape[1] - cfg.blockDim

        if (block is None) or (block):
            ratio = 4  # block dimension is (9x9)

            start_row += ratio
            end_row -= ratio
            start_col += ratio
            end_col -= ratio

            for row in range(start_row, end_row, ratio + 1):
                for col in range(start_col, end_col, ratio + 1):
                    blk = image.get_block([row, col])  # 25x25 get the block 25x25 from the central pixel

                    mm = self.feature.max_min(blk)
                    hsv = self.feature.mean_hsv(blk)
                    rgb = self.feature.mean_rgb(blk)
                    std = self.feature.standar_deviation(blk)
                    gab = self.feature.gabor_filter(blk)

                    set = [*mm, *hsv, *rgb, *std, gab]
                    pred = self.learning.predict([set])
                    image_classified[row - cfg.blockDim: row - cfg.blockDim + (ratio * 2) + 1,
                    col - cfg.blockDim: col - cfg.blockDim + (ratio * 2) + 1] = pred
            return image_classified
        else:
            for row in range(start_row, end_row):
                for col in range(start_col, end_col):
                    blk = image.get_block([row, col])  # 25x25 get the block 25x25 from the central pixel
                    mm = self.feature.max_min(blk)
                    hsv = self.feature.mean_hsv(blk)
                    rgb = self.feature.mean_rgb(blk)
                    std = self.feature.standar_deviation(blk)
                    gab = self.feature.gabor_filter(blk)

                    set = [*mm, *hsv, *rgb, *std, gab]
                    image_classified[row - cfg.blockDim, col - cfg.blockDim] = self.learning.predict([set])
            return image_classified

    # m1 es la imagen original, y m2 es la segmentada
    def compare_ground_truth(self, m1, m2):
        m1 = m1.astype(int)
        m2 = m2.astype(int)
        roc = np.zeros((4,)) # VP, FP, FN, VN
        for row in range(m1.shape[0]):
            for col in range(m1.shape[1]):
                val1 = m1[row][col]
                val2 = m2[row][col]

                if (val1 == val2):
                    if (val1 == 1):
                        roc[0] += 1
                    elif (val1 == 0):
                        roc[3] += 1
                elif (val1 > val2):
                    roc[2] += 1
                else:
                    roc[1] += 1
        return roc

    # Print accurate
    def save_accurate(self, data, c):
        print("Saving image")
        path = "/home/mrobot/Documentos/TFG/code/imagenes/test/"
        file, ext = str.split(data, '.')
        nI = str(cfg.nImage)
        nS = str(cfg.nSample)
        io.imsave(path + file + "_" + nI + "_" + nS + ".png", c)

    # save segmentation
    def save_segmentation(self, data, c, string):
        print("Saving segmentation")
        path = "/home/mrobot/Documentos/TFG/code/imagenes/unity_test/"
        file, ext = str.split(data, '.')
        io.imsave(path + file + "_" + string + ".png", c)

    # Calculates the mean-square error between two images
    def mean_squared_error(self, im1, im2):
        from skimage.measure import compare_mse
        return compare_mse(im1, im2)
