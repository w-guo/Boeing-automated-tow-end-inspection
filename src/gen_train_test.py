import os
import cv2
import fnmatch
import math
import random
import params
import numpy as np
from shutil import copy2
from sklearn.model_selection import train_test_split
from utils.img_augment import chooseRotate
from utils.pre_processing import RunningStats, pre_process

np.random.seed(2018)

# %% Generate training and testing sets

data_dir = '../data/'
image_files = fnmatch.filter(os.listdir(data_dir + 'src'), '*.jpg')
train_set, test_set = train_test_split(image_files,
                                       test_size=0.1,
                                       random_state=2018)

# create subfolders for training and testing sets
subfolder_list = [
    'train/src', 'train/label', 'train/prep', 'test/src', 'test/label',
    'test/predict', 'test/results'
]
for subfolder in subfolder_list:
    if not os.path.exists(data_dir + subfolder):
        os.makedirs(data_dir + subfolder)

# create testing set
for file in test_set:
    copy2(data_dir + 'src/' + file, data_dir + 'test/src')
    copy2(data_dir + 'label/' + file, data_dir + 'test/label')

image_num = 1000
image_each = image_num / len(train_set)

# create augmented training set
for file in train_set:
    count = 1
    src = cv2.imread(data_dir + 'src/' + file, 0)
    label = cv2.imread(data_dir + 'label/' + file, 0)
    while count < image_each:
        temp = np.random.random()
        # crop (no scale)
        begin_w = random.randint(0, params.w - params.img_w - 1)
        begin_h = random.randint(0, params.h - params.img_h - 1)
        src_sub = src[begin_h:begin_h + params.img_h, begin_w:begin_w +
                      params.img_w]
        label_sub = label[begin_h:begin_h + params.img_h, begin_w:begin_w +
                          params.img_w]
        # rotate/flip
        src_sub, label_sub = chooseRotate(src_sub, label_sub, temp)
        cv2.imwrite((data_dir + 'train/src/' + os.path.splitext(file)[0] +
                     '_%d.jpg' % count), src_sub)
        # relabel the sub-images with full pixels labeled as class 1 to class 0
        if np.sum(np.rint(label_sub / 255)) == params.img_w * params.img_h:
            label_sub = np.zeros_like(label_sub)
        cv2.imwrite((data_dir + 'train/label/' + os.path.splitext(file)[0] +
                     '_%d.jpg' % count), label_sub)
        count += 1

# %%  Pre-processing

rs = RunningStats()

train_file_list = fnmatch.filter(os.listdir(data_dir + 'train/src'), '*.jpg')
for file in train_file_list:
    img_rgb = cv2.imread(data_dir + 'train/src/' + file, 0)
    rs.push(img_rgb)

mean = rs.mean()
variance = rs.variance()
train_mean = np.mean(mean)
train_std = math.sqrt(
    np.sum(variance + np.multiply(mean, mean)) /
    (params.img_h * params.img_w) - train_mean**2)

np.save('./train_mean_std', np.array([train_mean, train_std]))

for file in train_file_list:
    img = cv2.imread(data_dir + 'train/src/' + file, 0)
    img_prep = pre_process(img, train_mean, train_std)
    cv2.imwrite(data_dir + 'train/prep/' + file, img_prep)
