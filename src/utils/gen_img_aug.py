import os
import cv2
import fnmatch
import random
import numpy as np
from shutil import copy2
from sklearn.model_selection import train_test_split

np.random.seed(2018)

img_w, img_h = (256, 256)
w, h = (968, 728)
folder = './aug_100k/'


def scale(src, label, factor):
    # scale in: scale image by given factor < 1
    # scale out: scale image by given factor > 1
    dst_img = cv2.resize(src, None, fx=factor, fy=factor,
                         interpolation=cv2.INTER_LINEAR)
    dst_label = cv2.resize(label, None, fx=factor,
                           fy=factor, interpolation=cv2.INTER_LINEAR)
    return dst_img, dst_label


def chooseScale(src, label, randTemp):
    if randTemp < 1/3:
        # scale in
        begin_w = random.randint(0, w-2*img_w-1)
        begin_h = random.randint(0, h-2*img_h-1)
        src_sub = src[begin_h: begin_h + 2*img_h, begin_w: begin_w + 2*img_w]
        label_sub = label[begin_h: begin_h +
                          2*img_h, begin_w: begin_w + 2*img_w]
        src_sub_l1, label_sub_l1 = scale(src_sub, label_sub, 0.5)
    elif 1/3 <= randTemp < 2/3:
        # no scale
        begin_w = random.randint(0, w-img_w-1)
        begin_h = random.randint(0, h-img_h-1)
        src_sub_l1 = src[begin_h: begin_h + img_h, begin_w: begin_w+img_w]
        label_sub_l1 = label[begin_h: begin_h + img_h, begin_w: begin_w+img_w]
    else:
        # scale out
        begin_w = random.randint(0, w-0.5*img_w-1)
        begin_h = random.randint(0, h-0.5*img_h-1)
        src_sub = src[begin_h: int(begin_h + 0.5*img_h),
                      begin_w: int(begin_w + 0.5*img_w)]
        label_sub = label[begin_h: int(
            begin_h + 0.5*img_h), begin_w: int(begin_w+0.5*img_w)]
        src_sub_l1, label_sub_l1 = scale(src_sub, label_sub, 2)
    return src_sub_l1, label_sub_l1


def chooseRotate(src_sub_l1, label_sub_l1, randTemp):

    src_sub_r90 = np.rot90(src_sub_l1)
    label_sub_r90 = np.rot90(label_sub_l1)
    if randTemp < 1/8:
        src_sub_l2 = src_sub_l1
        label_sub_l2 = label_sub_l1
    elif 1/8 <= randTemp < 2/8:
        # rotate 90 degrees
        src_sub_l2 = src_sub_r90
        label_sub_l2 = label_sub_r90
    elif 2/8 <= randTemp < 3/8:
        # rotate 180 degrees
        src_sub_l2 = np.rot90(src_sub_l1, 2)
        label_sub_l2 = np.rot90(label_sub_l1, 2)
    elif 3/8 <= randTemp < 4/8:
        # rotate 270 degrees
        src_sub_l2 = np.rot90(src_sub_l1, 3)
        label_sub_l2 = np.rot90(label_sub_l1, 3)
    elif 4/8 <= randTemp < 5/8:
        # flip image horizontally
        src_sub_l2 = cv2.flip(src_sub_l1, 0)
        label_sub_l2 = cv2.flip(label_sub_l1, 0)
    elif 5/8 <= randTemp < 6/8:
        # flip image vertically
        src_sub_l2 = cv2.flip(src_sub_l1, 1)
        label_sub_l2 = cv2.flip(label_sub_l1, 1)
    elif 6/8 <= randTemp < 7/8:
        # flip image horizontally
        src_sub_l2 = cv2.flip(src_sub_r90, 0)
        label_sub_l2 = cv2.flip(label_sub_r90, 0)
    else:
        # flip image vertically
        src_sub_l2 = cv2.flip(src_sub_r90, 1)
        label_sub_l2 = cv2.flip(label_sub_r90, 1)

    return src_sub_l2, label_sub_l2


def chooseEquHist(src_sub_l1, randTemp):
    if randTemp < 0.5:
        src_sub_l2 = cv2.equalizeHist(np.array(src_sub_l1, dtype=np.uint8))
    else:
        src_sub_l2 = src_sub_l1
    return src_sub_l2


gamma = 1.2
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) *
                  255 for i in np.arange(0, 256)]).astype("uint8")


def chooseGamma(src_sub_l1, randTemp):
    if randTemp < 0.5:
        src_sub_l2 = cv2.LUT(np.array(src_sub_l1, dtype=np.uint8), table)
    else:
        src_sub_l2 = src_sub_l1
    return src_sub_l2


def chooseGaussNoise(src_sub_l1, randTemp):
    if randTemp < 0.5:
        gauss = src_sub_l1.copy()
        cv2.randn(gauss, 0, 1)
        src_sub_l2 = src_sub_l1 + gauss
    else:
        src_sub_l2 = src_sub_l1
    return src_sub_l2


def chooseSPNoise(src_sub_l1, randTemp):
    src_sub_l2 = src_sub_l1.copy()
    salt_vs_pepper = 0.5
    amount = 0.004
    num_salt = np.ceil(amount * src_sub_l2.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * src_sub_l2.size * (1.0 - salt_vs_pepper))

    if randTemp < 1/2:
        # add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in src_sub_l2.shape]
        src_sub_l2[coords[0], coords[1]] = 255
        # add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in src_sub_l2.shape]
        src_sub_l2[coords[0], coords[1]] = 0
    else:
        src_sub_l2 = src_sub_l1
    return src_sub_l2


image_files = fnmatch.filter(os.listdir('./data/src'), '*.jpg')
train_set, test_set = train_test_split(
    image_files, test_size=0.1, random_state=2018)

for filename in test_set:
    copy2('./data/src/'+filename, folder+'test/src')
    copy2('./data/label/'+filename, folder+'test/label')

image_num = 100000
image_each = image_num / len(train_set)

for filename in train_set:
    count = 1
    src = cv2.imread('./data/src/'+filename, 0)
    label = cv2.imread('./data/label/'+filename, 0)
    while count < image_each:
        temp = np.random.random()
        # no scale
        src_sub, label_sub = chooseScale(src, label, 0.5)
        # rotate/flip
        src_sub, label_sub = chooseRotate(src_sub, label_sub, temp)
        cv2.imwrite((folder + 'train/src/' + filename +
                     '_%d.jpg' % count), src_sub)
        # relabel the sub-images with full pixels labeled as class 1 to class 0
        if np.sum(np.rint(label_sub/255)) == img_w*img_h:
            label_sub = np.zeros_like(label_sub)
        cv2.imwrite((folder+'train/label/'+filename +
                     '_%d.jpg' % count), label_sub)
        count += 1
