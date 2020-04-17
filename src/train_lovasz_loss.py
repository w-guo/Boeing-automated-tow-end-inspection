import sys
sys.path.insert(0, 'C:\\Users\\BARC\\Desktop\\AFP\\binary_seg')

import os, cv2, fnmatch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
from keras.optimizers import Adam, SGD
import tensorflow as tf
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
K.set_image_data_format('channels_last') 
from losses import bce_dice_loss, lovasz_loss, symmetric_lovasz_loss, iou, iou_lovasz
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from normalized_optimizer_wrapper import NormalizedOptimizer
from snapshot_update_per_epoch import SnapshotCallbackBuilder
#from snapshot_update_per_batch import SnapshotCallbackBuilder
np.random.seed(2018)
img_w, img_h = (256, 256) 

init_lr = 0.001
accum_it = 4.
filepath ='./train/'  

train_files = fnmatch.filter(os.listdir('./train/src'), '*.jpg')

def cov_to_class(val):    
    for i in range(0, 11):
        if val*10 <= i:
            return i
        
cov_class = np.zeros(len(train_files))
for i in range(len(train_files)):
    idx = train_files[i];
    label =  np.array(load_img('./train/label/'+idx, color_mode='grayscale'))/255
    coverage = np.sum(label)/(img_w*img_h)
    cov_class[i] = cov_to_class(coverage)  

#plt.style.use("ggplot")
##n, bins, _ = plt.hist(cov_class, 11, [-0.5,10.5])
#fig = plt.hist(cov_class, 11, [-0.5,10.5], label='class 1 coverage') 
#plt.xlabel("Coverage class")
#plt.ylabel("Frequency")
#plt.legend(loc="best")
#plt.savefig('./coverage_class.jpg', dpi=300) 
#plt.close()

train_set, val_set = train_test_split(train_files, test_size=0.2, stratify=cov_class, random_state=2018)

# data for training   
def generateData(batch_size,data=[]):  
    #print 'generateData...'
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for filename in data: 
            batch += 1 
            img = cv2.imread(filepath + 'prep/' + filename)
            img = img_to_array(img)/255  
            train_data.append(img)  
            label = cv2.imread(filepath + 'label/' + filename, cv2.IMREAD_GRAYSCALE)
            label = np.rint(label/255)
            label = img_to_array(label)
            train_label.append(label)  
            if batch % batch_size==0: 
                train_data = np.array(train_data)  
                train_label = np.array(train_label)  
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  
                
model = load_model('weights/unet_xception_resnet_nsgd32_best.h5',custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou})
#model = load_model('weights/unet_xception_resnet_nsgd32_lovasz_best.h5',custom_objects={'lovasz_loss': lovasz_loss, 'iou_lovasz': iou_lovasz})

input_x = model.layers[0].input
# remove layter activation layer and use losvasz loss
#output_layer = model.layers[-1].input
output_layer = model.layers[-1].output
model = Model(input_x, output_layer)

model_prefix = 'unet_xception_resnet_nsgd32_lovasz'
sgd = SGD(init_lr, momentum=0.9, nesterov=True, accum_iters=accum_it)
sgd = NormalizedOptimizer(sgd, normalization='l2')
# lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation  
# Then the default threshod for pixel prediction is 0 instead of 0.5
model.compile(loss=lovasz_loss, optimizer=sgd, metrics=[iou_lovasz])
model.summary()  

train_numb = len(train_set)  
valid_numb = len(val_set) 
print ("the number of train data is", train_numb)  
print ("the number of val data is", valid_numb)
n_epochs = 40
n_cycles = 1
BS = 8
snapshot = SnapshotCallbackBuilder(n_epochs, n_cycles, init_lr)
#snapshot = SnapshotCallbackBuilder(n_epochs, n_cycles, init_lr, steps_per_epoch=train_numb//(BS*accum_it))
H = model.fit_generator(generator=generateData(BS,train_set), steps_per_epoch=train_numb//(BS*accum_it)*accum_it, epochs=n_epochs, verbose=1,  
                validation_data=generateData(BS,val_set), validation_steps=valid_numb//(BS*accum_it)*accum_it, callbacks=snapshot.get_callbacks(model_prefix=model_prefix))  

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = n_epochs
plt.plot(np.arange(1, N+1), H.history["loss"], label="training loss")
plt.plot(np.arange(1, N+1), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(1, N+1), H.history["iou"], label="training IoU")
plt.plot(np.arange(1, N+1), H.history["val_iou"], label="validation IoU")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="best")
plt.savefig(model_prefix+'_iou.png', dpi=300)
plt.close()

plt.figure()
plt.plot(np.arange(1, N+1), H.history["loss"], label="training loss")
plt.plot(np.arange(1, N+1), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(1, N+1), H.history["acc"], label="training accuracy")
plt.plot(np.arange(1, N+1), H.history["val_acc"], label="validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/IoU")
plt.legend(loc="best")
plt.savefig(model_prefix+'.png', dpi=300)

