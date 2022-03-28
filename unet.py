#!/usr/bin/env python
# coding: utf-8

from importlib import import_module

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from shared_files.data_generators import *
from tabulate import tabulate
from shared_files.param import *

import gc
import glob
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import math
import pandas as pd
from PIL import Image
import numpy as np
import scipy.stats
import os
import random
from tensorflow.keras.losses import MAE as mae
from tensorflow.keras.losses import binary_crossentropy as bce
from tensorflow.keras.backend import flatten 
import sys
import segmentation_models as sm
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


gen = tr_gen(batch_size = 16*3,flip = True,crop = True)
v_gen = val_gen(batch_size =16*3)

def generator(g):
  while True:
    nex = next(g)[0]
    yield ((2*nex[-2]/255)-1),np.stack([nex[-1][...,0]+nex[-1][...,1],nex[-1][...,2],nex[-1][...,3]],axis = -1)

gen = generator(gen)
v_gen = generator(v_gen)

# CREATE UNET
model = sm.Unet('mobilenetv2',encoder_weights = None, classes=3,input_shape= (None,None,1), activation='softmax')
model.compile(
    'Adam',
    loss=sm.losses.categorical_focal_loss,
    metrics=[sm.metrics.FScore(class_weights = [0,1,0],name = 'GGO F1'),sm.metrics.FScore(class_weights = [0,0,1],name = 'CON F1'),sm.metrics.f1_score]
)

path = '/home/judah/Desktop/research/stichnet/cleaned_cropped_full_ct/train'
image_path_list = list(glob.iglob(path+'/**/*.jpg',recursive = True))
num_ims = len(image_path_list)

path = '/home/judah/Desktop/research/stichnet/cleaned_cropped_full_ct/val'
image_path_list = list(glob.iglob(path+'/**/*.jpg',recursive = True))
num_val_ims = len(image_path_list)

def pred():
        path = '/home/judah/Desktop/research/stichnet/cleaned_cropped_full_ct/train'
        train_path_list = list(glob.iglob(path+'/**/*.jpg',recursive = True))
        path = '/home/judah/Desktop/research/stichnet/cleaned_cropped_full_ct/val'
        val_path_list = list(glob.iglob(path+'/**/*.jpg',recursive = True))
        path = '/home/judah/Desktop/research/stichnet/cleaned_cropped_full_ct/test'
        test_path_list = list(glob.iglob(path+'/**/*.jpg',recursive = True))
        img_path_list = np.array(train_path_list + val_path_list + test_path_list)
#
        img_list = []
        msk_list = []
        for i in img_path_list:
          img = np.array(Image.open(i))
          msk = np.array(Image.open(i.replace('img','msk').replace('jpg','png')))
          aug = get_validation_augmentation(np.max(img.shape))(image = img,mask = msk)
          img_list.append(aug['image'])
          msk_list.append(aug['mask'])
#
        img = np.stack(img_list,axis = 0).astype('float32')
        print(img.max())
        img = (2*img/255)-1
        print(img.max())
        msk = np.stack(msk_list,axis = 0)
#
        pred_msk = model.predict(img,batch_size = 16*3)
#
        i = 0
        for path in img_path_list:
          pred_path = path.replace('img','pred/unet/').replace('jpg','png')
          m = pred_msk[i]
          m = np.argmax(m,axis = -1)
          #m = m - 1
          #m[m == -1] = 0
          m = get_one_hot(m,3)
          m = m*255
          m = m.astype('uint8')
          m = Image.fromarray(m)
          m.save(pred_path)
          i+=1
        return pred_msk,img

#model.fit(x=gen,
#          epochs = 500,
#          steps_per_epoch = int(num_ims // 16) + 1,
#          validation_data = v_gen,
#          validation_steps = int(num_val_ims // 16) + 1,
#          verbose = 1)
#model.save_weights('./weights/unet/unet_weights.tf')
model.load_weights('./weights/unet/unet_weights.tf')
pred_msk,img = pred() 
