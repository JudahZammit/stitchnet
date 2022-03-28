from shared_files.param import *

import glob
import albumentations as A
from PIL import Image
import random 
import os
import math
import numpy as np
import cv2

def get_training_augmentation(max_dim,crop = True,flip = False,light = False):
    train_transform = [A.PadIfNeeded(min_height=max_dim,min_width=max_dim,border_mode = 0)]
    train_transform.append(A.Resize(height = SHAPE, width = SHAPE, interpolation=1, always_apply=True, p=1))
    
    if crop:
        train_transform.append(A.RandomCrop(height=int(0.8*SHAPE),width=int(0.8*SHAPE),p=.5))
        train_transform.append(A.Resize(height = SHAPE, width = SHAPE, interpolation=1, always_apply=True, p=1))
            

    if flip:
        train_transform.append(A.VerticalFlip(p=.5)) 
        train_transform.append(A.RandomRotate90(p=.5)) 
    if light:
        train_transform.append(A.CLAHE(p=0.8)) 
        train_transform.append(A.RandomBrightnessContrast(p=0.8)) 
        train_transform.append(A.RandomGamma(p=0.8)) 
    return A.Compose(train_transform)


def get_validation_augmentation(max_dim):
    test_transform = [
        A.PadIfNeeded(min_height=max_dim,min_width=max_dim,border_mode = 0),
        A.Resize(height = SHAPE, width = SHAPE, interpolation=1, always_apply=True, p=1)
    ]
    return A.Compose(test_transform)


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[targets.reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def lab_img_gen(batch_size,path,shape = (SHAPE,SHAPE),crop = False,flip = False,light=False):
  
    image_path_list = list(glob.iglob(path+'/**/*.jpg',recursive = True))
     
    X_s = np.zeros((batch_size, shape[1], shape[0],RGB), dtype='float32')
    Y_s = np.zeros((batch_size, shape[1], shape[0],4), dtype='float32')

    def getitem(i):
        n = 0

        for x in image_path_list[i*batch_size:(i+1)*batch_size]:
            
            image = np.array(Image.open(x))
            msk = np.array(Image.open(x.replace('img','msk').replace('jpg','png')))
            max_dim = max(image.shape[0],image.shape[1]) 
            
            aug = get_training_augmentation(max_dim,crop = crop,flip = flip,light = light
                    )(image=image,mask = msk)
            image = aug['image']
            msk = aug['mask']
            msk = msk.astype('uint8')
            msk = get_one_hot(msk,4)
            X_s[n] = image[...,np.newaxis]
            Y_s[n] = msk
            n = n + 1
        return X_s,Y_s  
      
    def on_epoch_end():
        random.shuffle(image_path_list)

    i = -1
    while True :
        if i < len(image_path_list) // batch_size:
            i = i + 1
        else:
            on_epoch_end()
            i = 0
        yield getitem(i)

def img_gen(batch_size,path,shape = (SHAPE,SHAPE),crop = False,flip = False,light=False):
  
    image_path_list = list(glob.iglob(path+'/**/*.jpg',recursive = True))
  
    X_s = np.zeros((batch_size, shape[1], shape[0],RGB), dtype='float32')

    def getitem(i):
        n = 0

        for x in image_path_list[i*batch_size:(i+1)*batch_size]:
            
            image = np.array(Image.open(x))
            max_dim = max(image.shape[0],image.shape[1]) 
            
            aug = get_training_augmentation(max_dim,crop = crop,flip = flip,light = light
                    )(image=image)
            image = aug['image']
            X_s[n] = image[...,np.newaxis]
            n = n + 1
        return X_s  
      
    def on_epoch_end():
        random.shuffle(image_path_list)

    i = -1
    while True :
        if i < len(image_path_list) // batch_size:
            i = i + 1
        else:
            on_epoch_end()
            i = 0
        yield getitem(i)

def tr_gen(batch_size,shape = (SHAPE,SHAPE),
        crop = False,flip = False,light = False):
    
    base_hel = '/home/judah/Desktop/research/stichnet/cleaned_cropped_full_ct/unlabelled/healthy'
    hel_gen = img_gen(batch_size//3,base_hel,shape,crop,flip,light)
    
    base_ncp = '/home/judah/Desktop/research/stichnet/cleaned_cropped_full_ct/unlabelled/ncp'
    ncp_gen = img_gen(batch_size//3,base_ncp,shape,crop,flip,light)
    
    base_lab = '/home/judah/Desktop/research/stichnet/cleaned_cropped_full_ct/train'
    lab_gen = lab_img_gen(batch_size//3,base_lab,shape,crop,flip,light)
 
    T_hel = np.zeros((batch_size//3, shape[1], shape[0],RGB), dtype='float32') 
    Y_hel = np.zeros((batch_size//3, shape[1], shape[0],4), dtype='float32') 
    T_ncp = np.zeros((batch_size//3, shape[1], shape[0],RGB), dtype='float32') 
    T_lab = np.zeros((batch_size//3, shape[1], shape[0],RGB), dtype='float32') 
    Y_lab = np.zeros((batch_size//3, shape[1], shape[0],4), dtype='float32') 
 
    while True :
        
        t = next(hel_gen)
        T_hel[:,:,:,:] = t[:,:,:,:]
        Y_hel[:,:,:,:] = 0
        Y_hel[...,0][T_hel[...,0] == 0] = 1
        Y_hel[...,1][T_hel[...,0] != 0] = 1
                
        t = next(ncp_gen)
        T_ncp[:,:,:,:] = t[:,:,:,:]

        t,y = next(lab_gen)
        T_lab[:,:,:,:] = t[:,:,:,:]
        Y_lab[:,:,:,:] = y[:,:,:,:]


        yield ((T_hel,Y_hel,T_ncp,T_lab,Y_lab),{})

def val_gen(batch_size,shape = (SHAPE,SHAPE),
        crop = False,flip = False,light = False):
    
    base_hel = '/home/judah/Desktop/research/stichnet/cleaned_cropped_full_ct/unlabelled/healthy'
    hel_gen = img_gen(batch_size//3,base_hel,shape,crop,flip,light)
    
    base_ncp = '/home/judah/Desktop/research/stichnet/cleaned_cropped_full_ct/unlabelled/ncp'
    ncp_gen = img_gen(batch_size//3,base_ncp,shape,crop,flip,light)
    
    base_lab = '/home/judah/Desktop/research/stichnet/cleaned_cropped_full_ct/val'
    lab_gen = lab_img_gen(batch_size//3,base_lab,shape,crop,flip,light)
 
    T_hel = np.zeros((batch_size//3, shape[1], shape[0],RGB), dtype='float32') 
    Y_hel = np.zeros((batch_size//3, shape[1], shape[0],4), dtype='float32') 
    T_ncp = np.zeros((batch_size//3, shape[1], shape[0],RGB), dtype='float32') 
    T_lab = np.zeros((batch_size//3, shape[1], shape[0],RGB), dtype='float32') 
    Y_lab = np.zeros((batch_size//3, shape[1], shape[0],4), dtype='float32') 
 
    while True :
        
        t = next(hel_gen)
        T_hel[:,:,:,:] = t[:,:,:,:]
        Y_hel[:,:,:,:] = 0
        Y_hel[...,0][T_hel[...,0] == 0] = 1
        Y_hel[...,1][T_hel[...,0] != 0] = 1
                
        t = next(ncp_gen)
        T_ncp[:,:,:,:] = t[:,:,:,:]

        t,y = next(lab_gen)
        T_lab[:,:,:,:] = t[:,:,:,:]
        Y_lab[:,:,:,:] = y[:,:,:,:]


        yield ((T_hel,Y_hel,T_ncp,T_lab,Y_lab),{})




