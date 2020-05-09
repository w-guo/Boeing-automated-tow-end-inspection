import os
import cv2
import fnmatch
import params
import numpy as np
from keras.models import load_model
from utils.losses import lovasz_loss, iou_lovasz
from utils.img_reconstruction import rotate_mirror_do, rotate_mirror_undo, windowed_subdivs, recreate_from_subdivs
from tqdm import tqdm

# %% Reconstruct test images from predicted sub-images

# load the saved model from the 2nd stage training
model = load_model('weights/unet_xception_resnet_nsgd32_lovasz_best.h5',
                   custom_objects={
                       'lovasz_loss': lovasz_loss,
                       'iou_lovasz': iou_lovasz
                   })

train_mean, train_std = np.load('train_mean_std.npy')

test_dir = '../data/test/'
test_file_list = fnmatch.filter(os.listdir(test_dir + 'src/'), '*.jpg')
for file in test_file_list:
    img = cv2.imread(test_dir + 'src/' + file, 0)
    pad = np.pad(img, pad_width=params.borders, mode='reflect')
    pads = rotate_mirror_do(pad)
    res = []
    for pad in tqdm(pads):
        # for every rotation:
        sd = windowed_subdivs(model, 3, train_mean, train_std, pad,
                              params.window_size, params.overlap_pct)
        one_padded_result = recreate_from_subdivs(sd,
                                                  params.window_size,
                                                  params.overlap_pct,
                                                  padded_out_shape=list(
                                                      pad.shape))
        res.append(one_padded_result)
    # merge after rotations:
    padded_results = rotate_mirror_undo(res)
    # convert the output of a model with lovasz loss as the metric to probability
    prob = np.exp(padded_results) / (1 + np.exp(padded_results))
    prd = prob[params.aug_h:params.aug_h +
               params.h, params.aug_w:params.aug_w + params.w]
    cv2.imwrite(test_dir + 'predict/' + os.path.splitext(file)[0] + '.png',
                np.rint(prd * 255))
