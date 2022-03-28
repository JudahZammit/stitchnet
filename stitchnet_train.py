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

modelName = sys.argv[1]

mod = import_module('{}.layers.filler'.format(modelName),)
myModel = mod.myModel

mod = import_module('{}.layers.state'.format(modelName))
State = mod.State
l = State.layers

MODEL_PATH = './{}/'.format(modelName)
WEIGHT_PATH = MODEL_PATH + 'weights.tf'

gen = tr_gen(batch_size = BS,crop = True,flip = True)
v_gen = val_gen(batch_size =BS)
model = myModel()
model_opt = tf.keras.optimizers.Adam(lr=.0003,clipnorm = 1.,clipvalue = 0.5)
model.compile(model_opt)


path = '/home/judah/Desktop/research/stichnet/cleaned_cropped_full_ct/unlabelled'
image_path_list = list(glob.iglob(path+'/**/*.jpg',recursive = True))
num_ims = len(image_path_list)


# Reads in all the labelled images, uses stitchnet to predict their segmentations and writes the predictions to disk
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
        img = np.stack(img_list,axis = 0)
        msk = np.stack(msk_list,axis = 0)
#
        pred_msk = []
        dummy = list(next(gen)[0])
        for i in range(img.shape[0]):
          dummy[-2] = np.expand_dims(np.expand_dims(img[i].astype('float32'),axis = -1),axis = 0)
          out = model(dummy,gen = False,training = True)
          pred_msk.append(out['y_reconstructed'][-1])
#
        pred_msk = np.stack(pred_msk,axis = 0)
#        
#
        i = 0
        for path in img_path_list:
          pred_path = path.replace('img','pred/gt').replace('jpg','png')
          m = msk[i].astype('float32')
          m = m - 1
          m[m == -1] = 0
          m = get_one_hot(m.astype('uint8'),3)
          m = m*255
          m = m.astype('uint8')
          m = Image.fromarray(m)
          m.save(pred_path)
          i+=1

def setKL(kl):
  tf.keras.backend.set_value(l['KL'], kl)
  
# Trains the StitchNet Model
def train():
        model.save_weights(MODEL_PATH+'fresh_weights/weights.tf')
        #model.load_weights(MODEL_PATH+'weights.tf')
        setKL(1.0)
        model.fit(x=gen,
                  epochs = 80,
                  steps_per_epoch = int(num_ims/BS)+1,
                  validation_data = v_gen,
                  validation_steps = (150*3 // BS) + 1,
                  verbose = 1)
        model.save_weights(MODEL_PATH+'weights.tf')

#train()
#pred()
model.load_weights('./weight/stitchnet/stitchnet_weights.tf')
fig_pred()
