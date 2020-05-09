import cv2
import math
import numpy as np


class RunningStats:
    """
    Welford's online algorithm to update mean and (estimated) variance of streaming data
    https://github.com/liyanage/python-modules/blob/master/running_stats.py
    """
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / self.n if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())


def adjust_gamma(img, gamma=1.0):
    """
    Build a lookup table mapping the pixel values [0, 255] to their adjusted 
    gamma values
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0)**invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_img = cv2.LUT(np.array(img, dtype=np.uint8), table)

    return new_img


def pre_process(img, train_mean, train_std):

    img_normalized = (img - train_mean) / train_std
    img_normalized = ((img_normalized - np.min(img_normalized)) /
                      (np.max(img_normalized) - np.min(img_normalized))) * 255
    img_equ = cv2.equalizeHist(np.array(img_normalized, dtype=np.uint8))
    img_prep = adjust_gamma(img_equ, 1.2)

    return img_prep
