from util.image import Image
from scipy.misc import imread
import numpy as np
import config as cfg
import time

"""
Implements methods to classify images
"""


def classify(melanoma, ground, feature, classifier, block=True):
    seg = []
    tim = []
    dim = []

    for (melanoma_item, ground_item) in zip(melanoma, ground):
        print('Segmentating...')
        print('\t'+melanoma_item)
        img = Image(cfg.melanoma_path + melanoma_item, cfg.ground_path + ground_item, cfg.block)
        size = img.get_shape()
        portion = img.get_portion()

        dim.append(size)

        img_seg = np.zeros((size[0], size[1]))

        row = [portion, size[0] - portion]
        col = [portion, size[1] - portion]

        if block:
            st = time.time()
            seg.append(per_block(img, img_seg, row, col, feature, classifier))
            tim.append(time.time() - st)
        else:
            st = time.time()
            seg.append(per_pixel(img, img_seg, row, col, feature, classifier))
            tim.append(time.time() - st)

    return seg, tim, dim


def per_block(img, img_seg, row, col, feature, classifier):
    ratio = 4
    adv = 2 * ratio + 1

    for r in range(row[0], row[1], adv):
        for c in range(col[0], col[1], adv):
            blk = img.get_block([r, c])
            val = feature.features(blk)
            pre = classifier.predict([val])
            img_seg[r: r + adv, c: c + adv] = pre
    return img_seg


def per_pixel(img, img_seg, row, col, feature, classifier):
    for r in range(row[0], row[1]):
        for c in range(col[0], col[1]):
            blk = img.get_block([r, c])
            val = feature.features(blk)
            pre = classifier.predict([val])
            img_seg[r, c] = pre
    return img_seg


def local_error(confmat):
    """
    Calculates the accuracy by each image

    Parameters
    ----------
    confmat: list of lists
        The confusion matrix to calculate the accuracy, TP, FP, FN, TN

    Returns
    -------
        A list of list with 3 values (Sensitivity, Specificity, Accuracy)
    """
    local_err = []
    for mat in confmat:
        TP = mat[0]
        FP = mat[1]
        FN = mat[2]
        TN = mat[3]
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        local_err.append([sensitivity, specificity, accuracy])
    return local_err


def total_error(local_acc):
    """
    Calculates the mean accuracy of a list of local accuracys

    Parameters
    ----------
    local_acc: list of lists
        The local accuracy of each image

    Returns
    -------
        3 values, (Sensitivity, Specificity, Accuracy)
    """
    acc = np.array(local_acc)
    sensitivity_mean = acc[:, 0].mean()
    sensitivity_std = acc[:, 0].std()

    specificity_mean = acc[:, 1].mean()
    specificity_std = acc[:, 1].std()

    accuracy_mean = acc[:, 2].mean()
    accuracy_std = acc[:, 2].std()

    return [sensitivity_mean, sensitivity_std], [specificity_mean, specificity_std], [accuracy_mean, accuracy_std]


def estimate_error(confmat):
    """
    Calculates the accuracy from a confusion matrix.

    Parameters
    ----------
    confmat: list of lists
        The confusion matrix to calculate the accuracy

    Returns
    -------
        3 values, (Sensitivity, Specificity, Accuracy)
    """
    cm = np.array(confmat)
    TP = cm[:, 0].sum()
    FP = cm[:, 1].sum()
    FN = cm[:, 2].sum()
    TN = cm[:, 3].sum()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + FP + TN + FN)

    return sensitivity, specificity, accuracy


def confusion_matrix(seg, ground_list):
    """
    Calculates the confusion matrix

    Parameters
    ----------
    seg: list
        A list with the segmented images. Each item of the list is a 2D-array
    ground_list: list
        A list with all the ground_truth path of the segmented images.

    Returns
    -------
    A list of 1D-array. Each sublist contents the values TP, FP, FN, TN of a confusion matrix.
    """
    from skimage import io

    acc = []

    for s, g in zip(seg, ground_list):
        ground = imread(cfg.ground_path + g)

        m1 = ground.astype(int)
        m2 = s.astype(int)

        conf = np.zeros((4,), dtype=int) # TP, FP, FN, TN
        for row in range(m1.shape[0]):
            for col in range(m1.shape[1]):
                val1 = m1[row][col]
                val2 = m2[row][col]

                if val1 == val2:
                    if val1 == 255:
                        conf[0] += 1
                    elif val1 == 0:
                        conf[3] += 1
                elif val1 > val2:
                    conf[2] += 1
                else:
                    conf[1] += 1
        acc.append(conf)
    return acc

"""
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
"""