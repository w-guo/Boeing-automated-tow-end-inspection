import os
import cv2
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import params as prm
from sklearn.model_selection import train_test_split
from model import build_model
from utils.snapshot_callback import SnapshotCallbackBuilder
from utils.normalized_optimizer_wrapper import NormalizedOptimizer
from utils.losses import bce_dice_loss, lovasz_loss, iou, iou_lovasz
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model, load_model
from keras.optimizers import SGD


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


def generateData(batch_size, data=[]):
    while True:
        train_data = []
        train_label = []
        batch = 0
        for filename in data:
            batch += 1
            img = cv2.imread(train_dir + 'prep/' + filename)
            img = img_to_array(img) / 255
            train_data.append(img)
            label = cv2.imread(train_dir + 'label/' + filename,
                               cv2.IMREAD_GRAYSCALE)
            label = np.rint(label / 255)
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0


np.random.seed(2018)

# %% Split training and validation sets

train_dir = '../data/train/'
train_file_list = fnmatch.filter(os.listdir(train_dir + 'src'), '*.jpg')
cov_class = np.zeros(len(train_file_list))
for i in range(len(train_file_list)):
    idx = train_file_list[i]
    label = np.array(
        load_img(train_dir + 'label/' + idx, color_mode='grayscale')) / 255
    coverage = np.sum(label) / (prm.img_w * prm.img_h)
    cov_class[i] = cov_to_class(coverage)
train_set, val_set = train_test_split(train_file_list,
                                      test_size=0.2,
                                      stratify=cov_class,
                                      random_state=2018)

train_numb = len(train_set)
val_numb = len(val_set)

# %% 1st stage training

# training parameters
BS = 8  # batch size
accum_it = 4  # accumulated batches
init_lr = 0.001  # initial learning rate
n_epochs = 40
n_cycles = 1

model_1_prefix = 'unet_xception_resnet_nsgd32'
model_1 = build_model(16)
sgd = SGD(init_lr, momentum=0.9, nesterov=True, accum_iters=accum_it)
sgd = NormalizedOptimizer(sgd, normalization='l2')
model_1.compile(loss=bce_dice_loss, optimizer=sgd, metrics=[iou, 'accuracy'])
model_1.summary()

snapshot = SnapshotCallbackBuilder(n_epochs, n_cycles, init_lr)
H = model_1.fit_generator(
    generator=generateData(BS, train_set),
    steps_per_epoch=train_numb // (BS * accum_it) * accum_it,
    epochs=n_epochs,
    verbose=1,
    validation_data=generateData(BS, val_set),
    validation_steps=val_numb // (BS * accum_it) * accum_it,
    callbacks=snapshot.get_callbacks(model_prefix=model_1_prefix))

# %% 2nd stage training

# load the saved model from the 1st stage training
model_1 = load_model('weights/%s_best.h5' % model_1_prefix,
                     custom_objects={
                         'bce_dice_loss': bce_dice_loss,
                         'iou': iou
                     })

input_x = model_1.layers[0].input
# remove layter activation layer and use Losvasz loss
output_layer = model_1.layers[-1].output
model_2 = Model(input_x, output_layer)

model_2_prefix = model_1_prefix + '_lovasz'

# Lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation
# Then the default threshod for pixel prediction is 0 instead of 0.5
model_2.compile(loss=lovasz_loss, optimizer=sgd, metrics=[iou_lovasz])

H = model_2.fit_generator(
    generator=generateData(BS, train_set),
    steps_per_epoch=train_numb // (BS * accum_it) * accum_it,
    epochs=n_epochs,
    verbose=1,
    validation_data=generateData(BS, val_set),
    validation_steps=val_numb // (BS * accum_it) * accum_it,
    callbacks=snapshot.get_callbacks(model_prefix=model_2_prefix))

# %% Plot the training loss and IoU

plt.style.use("ggplot")
plt.figure()
plt.plot(H.epoch, H.history["loss"], label="training loss")
plt.plot(H.epoch, H.history["val_loss"], label="validation loss")
plt.plot(H.epoch, H.history["iou_lovasz"], label="training IoU")
plt.plot(H.epoch, H.history["val_iou_lovasz"], label="validation IoU")
plt.xlabel("Epoch")
plt.ylabel("Loss/IoU")
plt.legend(loc="best")
plt.savefig(model_2_prefix + '.png', dpi=300)