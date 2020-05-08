import cv2
import numpy as np

def chooseRotate(src_sub_l1, label_sub_l1, randTemp):

    src_sub_r90 = np.rot90(src_sub_l1)
    label_sub_r90 = np.rot90(label_sub_l1)
    if randTemp < 1/8:
        src_sub_l2 = src_sub_l1
        label_sub_l2 = label_sub_l1
    elif randTemp < 2/8:
        # rotate 90 degrees
        src_sub_l2 = src_sub_r90
        label_sub_l2 = label_sub_r90
    elif randTemp < 3/8:
        # rotate 180 degrees
        src_sub_l2 = np.rot90(src_sub_l1, 2)
        label_sub_l2 = np.rot90(label_sub_l1, 2)
    elif randTemp < 4/8:
        # rotate 270 degrees
        src_sub_l2 = np.rot90(src_sub_l1, 3)
        label_sub_l2 = np.rot90(label_sub_l1, 3)
    elif randTemp < 5/8:
        # flip image horizontally
        src_sub_l2 = cv2.flip(src_sub_l1, 0)
        label_sub_l2 = cv2.flip(label_sub_l1, 0)
    elif randTemp < 6/8:
        # flip image vertically
        src_sub_l2 = cv2.flip(src_sub_l1, 1)
        label_sub_l2 = cv2.flip(label_sub_l1, 1)
    elif randTemp < 7/8:
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

