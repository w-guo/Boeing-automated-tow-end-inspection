import fnmatch
import cv2
import os
from utils import rotate_mirror_do, rotate_mirror_undo, windowed_subdivs, recreate_from_subdivs
from normalized_optimizer_wrapper import NormalizedOptimizer
from losses import bce_dice_loss, lovasz_loss, symmetric_lovasz_loss, iou, iou_lovasz
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, UpSampling2D, Dropout, Lambda, Activation, Add
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import sys
sys.path.insert(0, 'C:\\Users\\BARC\\Desktop\\AFP\\binary_seg')

K.set_image_data_format('channels_last')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


test_src_files = fnmatch.filter(os.listdir('./test/src/'), '*.jpg')
# load the saved model from the 2nd stage training
model = load_model('weights/unet_xception_resnet_nsgd32_lovasz_best.h5',
                   custom_objects={'lovasz_loss': lovasz_loss, 'iou_lovasz': iou_lovasz})

train_mean, train_std = np.load('train_mean_std.npy')

img_w, img_h = (256, 256)
h, w = (728, 968)
overlap_pct = 0
window_size = 256
n_w = (w-img_w*overlap_pct)//(img_w*(1-overlap_pct))+1
n_h = (h-img_h*overlap_pct)//(img_h*(1-overlap_pct))+1

aug_w = int((img_w*(1-overlap_pct)*n_w+img_w*overlap_pct-w)/2)
aug_h = int((img_h*(1-overlap_pct)*n_h+img_h*overlap_pct-h)/2)
borders = ((aug_h, aug_h), (aug_w, aug_w))

if not os.path.exists('./test/predict/'):
    os.makedirs('./test/predict/')

for filename in test_src_files:
    img = cv2.imread('./test/src/' + filename, 0)
    pad = np.pad(img, pad_width=borders, mode='reflect')
    pads = rotate_mirror_do(pad)
    res = []
    for pad in tqdm(pads):
        # for every rotation:
        sd = windowed_subdivs(model, 3, train_mean,
                              train_std, pad, window_size, overlap_pct)
        one_padded_result = recreate_from_subdivs(
            sd, window_size, overlap_pct, padded_out_shape=list(pad.shape))
        res.append(one_padded_result)
    # merge after rotations:
    padded_results = rotate_mirror_undo(res)
    # convert the output of a model with lovasz loss as the metric to probability
    prob = np.exp(padded_results)/(1+np.exp(padded_results))
    prd = prob[aug_h:aug_h+h, aug_w:aug_w+w]
    cv2.imwrite('./test/predict/' + os.path.splitext(filename)
                [0]+'.png', np.rint(prd*255))
