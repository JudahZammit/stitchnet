from classification_models.tfkeras import Classifiers
from shared_files.param import *

import tensorflow.keras.layers as tfkl
import tensorflow.keras as tfk


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,ReLU, BatchNormalization, Add
from tensorflow.keras import layers
import numpy as np

class LowDecoderTransposeX2Block(layers.Layer):
    def __init__(self,filters,use_batchnorm=False,out = 'relu'):
        super(LowDecoderTransposeX2Block,self).__init__()
        
        self.l1 = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same'
        )
        self.l1_b = layers.BatchNormalization()
        self.l1_r = layers.Activation('relu') 
          
    def call(self,inputs):
        x = inputs
        
        x = self.l1(x)
        x = self.l1_b(x)
        x = self.l1_r(x)
        
        return x


class LowResBlock(layers.Layer):
  def __init__(self,filters):
        super(LowResBlock,self).__init__()
         
        self.r2_0 = layers.BatchNormalization()
        self.r2_1 = ReLU()
        self.r2_2 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters,padding = 'same')
        self.r2_3 = layers.BatchNormalization()
        self.r2_4 = ReLU()
        self.r2_5 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters,padding = 'same')
        self.r2_6 = Add()
        
        
  def call(self,inputs):
        x = inputs   
        
        x_shortcut = x   
        x = self.r2_0(x_shortcut)
        x = self.r2_1(x)
        x = self.r2_2(x)
        x = self.r2_3(x)
        x = self.r2_4(x)
        x = self.r2_5(x)
        x = self.r2_6([x,x_shortcut])
       
        return x        

class LowSkelaSmooth(layers.Layer):
  def __init__(self,filters):
        super(LowSkelaSmooth,self).__init__()
        
        self.r0_0 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters,padding = 'same')
        self.r0_1 = layers.BatchNormalization()
        self.r0_2 = layers.Activation('relu') 
        self.r0_3 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters,padding = 'same')
        
        self.o_b = layers.BatchNormalization()
        self.o_r = ReLU()
        
        
  def call(self,inputs):
        x = inputs   
        
        x = self.r0_0(x)
        x = self.r0_1(x)
        x = self.r0_2(x)
        x = self.r0_3(x)
        
        x = self.o_b(x)
        x = self.o_r(x)
       
        return x        

class LowLightSmooth(layers.Layer):
  def __init__(self,filters):
        super(LowLightSmooth,self).__init__()
         
        self.r0_0 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters,padding = 'same')
        self.r0_1 = layers.BatchNormalization()
        self.r0_2 = layers.Activation('relu') 
        self.r0_3 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters,padding = 'same')
 
        #ResNet Block
        self.r1 = LowResBlock(filters)       

        self.o_b = layers.BatchNormalization()
        self.o_r = ReLU()
       
 
  def call(self,inputs):
        x = inputs   
        
        x = self.r0_0(x)
        x = self.r0_1(x)
        x = self.r0_2(x)
        x = self.r0_3(x)
    
        x = self.r1(x)       
 
        x = self.o_b(x)
        x = self.o_r(x)
 
        return x        

class LowSmooth(layers.Layer):
  def __init__(self,filters):
        super(LowSmooth,self).__init__()
        
        self.r0_0 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters,padding = 'same')
        self.r0_1 = layers.BatchNormalization()
        self.r0_2 = layers.Activation('relu') 
        self.r0_3 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters,padding = 'same')
        
        self.r1 = LowResBlock(filters)
        self.r2 = LowResBlock(filters)
        self.r3 = LowResBlock(filters)

        self.o_b = layers.BatchNormalization()
        self.o_r = ReLU()

  def call(self,inputs):
        x = inputs   
      
        x = self.r0_0(x)
        x = self.r0_1(x)
        x = self.r0_2(x)
        x = self.r0_3(x)
        
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
    
        x = self.o_b(x)
        x = self.o_r(x)
        
        return x        

class LowDown(layers.Layer):
  def __init__(self,filters):
    super(LowDown,self).__init__()
        
    self.r0_0 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters,padding = 'same')
    self.r0_1 = layers.BatchNormalization()
    self.r0_2 = layers.Activation('relu') 

    
  def call(self,inputs):
    x = inputs   
      
    x = self.r0_0(x)
    x = self.r0_1(x)
    x = self.r0_2(x)
        
    return x        

class LowUp(layers.Layer):
  def __init__(self,filters):
    super(LowUp,self).__init__()
        
    self.r0_0 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters,padding = 'same')
    self.r0_1 = layers.BatchNormalization()
    self.r0_2 = layers.Activation('relu') 

    
  def call(self,inputs):
    x = inputs   
      
    x = self.r0_0(x)
    x = self.r0_1(x)
    x = self.r0_2(x)
        
    return x 
       
class HighDecoderTransposeX2Block(layers.Layer):
    def __init__(self,filters,use_batchnorm=False,out = 'relu'):
        super(HighDecoderTransposeX2Block,self).__init__()

        self.l0 = layers.Conv2D(kernel_size = 1,strides = 1,filters = filters//4,padding = 'same')
        self.l0_b = layers.BatchNormalization()
        self.l0_r = layers.Activation('relu') 
        
        self.l1 = layers.Conv2DTranspose(
            filters//4,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same'
        )
        self.l1_b = layers.BatchNormalization()
        self.l1_r = layers.Activation('relu') 
        
        self.l2 = layers.Conv2D(kernel_size = 1,strides = 1,filters = filters,padding = 'same')
        self.l2_b = layers.BatchNormalization()
        self.l2_r = layers.Activation('relu') 
   
  
    def call(self,inputs):
        x = inputs
          
        x = self.l0(x)
        x = self.l0_b(x)
        x = self.l0_r(x)
        
        x = self.l1(x)
        x = self.l1_b(x)
        x = self.l1_r(x)
        
        x = self.l2(x)
        x = self.l2_b(x)
        x = self.l2_r(x)

        return x


class HighResBlock(layers.Layer):
  def __init__(self,filters):
        super(HighResBlock,self).__init__()
         
        self.r2_0 = layers.BatchNormalization()
        self.r2_1 = ReLU()
        self.r2_2 = layers.Conv2D(kernel_size = 1,strides = 1,filters = filters//4,padding = 'same')
        self.r2_3 = layers.BatchNormalization()
        self.r2_4 = ReLU()
        self.r2_5 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters//4,padding = 'same')
        self.r2_6 = layers.BatchNormalization()
        self.r2_7 = ReLU()
        self.r2_8 = layers.Conv2D(kernel_size = 1,strides = 1,filters = filters,padding = 'same')
        self.r2_9 = Add()
        
        
  def call(self,inputs):
        x = inputs   
        
        x_shortcut = x   
        x = self.r2_0(x_shortcut)
        x = self.r2_1(x)
        x = self.r2_2(x)
        x = self.r2_3(x)
        x = self.r2_4(x)
        x = self.r2_5(x)
        x = self.r2_6(x)
        x = self.r2_7(x)
        x = self.r2_8(x)
        x = self.r2_9([x,x_shortcut])
       
        return x        

class HighSkelaSmooth(layers.Layer):
  def __init__(self,filters):
        super(HighSkelaSmooth,self).__init__()
        
        self.r0_0 = layers.Conv2D(kernel_size = 1,strides = 1,filters = filters//4,padding = 'same')
        self.r0_1 = layers.BatchNormalization()
        self.r0_2 = layers.Activation('relu') 
        self.r0_3 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters//4,padding = 'same')
        self.r0_4 = layers.BatchNormalization()
        self.r0_5 = layers.Activation('relu') 
        self.r0_6 = layers.Conv2D(kernel_size = 1,strides = 1,filters = filters,padding = 'same')
        
        self.o_b = layers.BatchNormalization()
        self.o_r = ReLU()
        
        
  def call(self,inputs):
        x = inputs   
        
        x = self.r0_0(x)
        x = self.r0_1(x)
        x = self.r0_2(x)
        x = self.r0_3(x)
        x = self.r0_4(x)
        x = self.r0_5(x)
        x = self.r0_6(x)
        
        x = self.o_b(x)
        x = self.o_r(x)
       
        return x        

class HighLightSmooth(layers.Layer):
  def __init__(self,filters):
        super(HighLightSmooth,self).__init__()
         
        self.r0_0 = layers.Conv2D(kernel_size = 1,strides = 1,filters = filters//4,padding = 'same')
        self.r0_1 = layers.BatchNormalization()
        self.r0_2 = layers.Activation('relu') 
        self.r0_3 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters//4,padding = 'same')
        self.r0_4 = layers.BatchNormalization()
        self.r0_5 = layers.Activation('relu') 
        self.r0_6 = layers.Conv2D(kernel_size = 1,strides = 1,filters = filters,padding = 'same')
 
        #ResNet Block
        self.r1 = ResBlock(filters)       

        self.o_b = layers.BatchNormalization()
        self.o_r = ReLU()
       
 
  def call(self,inputs):
        x = inputs   
        
        x = self.r0_0(x)
        x = self.r0_1(x)
        x = self.r0_2(x)
        x = self.r0_3(x)
        x = self.r0_4(x)
        x = self.r0_5(x)
        x = self.r0_6(x)
       
    
        x = self.r1(x)       
 
        x = self.o_b(x)
        x = self.o_r(x)
 
        return x        

class HighSmooth(layers.Layer):
  def __init__(self,filters):
        super(HighSmooth,self).__init__()
        
        self.r0_0 = layers.Conv2D(kernel_size = 1,strides = 1,filters = filters//4,padding = 'same')
        self.r0_1 = layers.BatchNormalization()
        self.r0_2 = layers.Activation('relu') 
        self.r0_3 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters//4,padding = 'same')
        self.r0_4 = layers.BatchNormalization()
        self.r0_5 = layers.Activation('relu') 
        self.r0_6 = layers.Conv2D(kernel_size = 1,strides = 1,filters = filters,padding = 'same')
        
        self.r1 = ResBlock(filters)
        self.r2 = ResBlock(filters)
        self.r3 = ResBlock(filters)

        self.o_b = layers.BatchNormalization()
        self.o_r = ReLU()

  def call(self,inputs):
        x = inputs   
      
        x = self.r0_0(x)
        x = self.r0_1(x)
        x = self.r0_2(x)
        x = self.r0_3(x)
        x = self.r0_4(x)
        x = self.r0_5(x)
        x = self.r0_6(x)
        
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
    
        x = self.o_b(x)
        x = self.o_r(x)
        
        return x        

class HighDown(layers.Layer):
  def __init__(self,filters):
    super(HighDown,self).__init__()
        
    self.r0_0 = layers.Conv2D(kernel_size = 1,strides = 1,filters = filters,padding = 'same')
    self.r0_1 = layers.BatchNormalization()
    self.r0_2 = layers.Activation('relu') 
    self.r0_3 = layers.Conv2D(kernel_size = 3,strides = 1,filters = filters,padding = 'same')
    self.r0_4 = layers.BatchNormalization()
    self.r0_5 = layers.Activation('relu') 

    
  def call(self,inputs):
    x = inputs   
      
    x = self.r0_0(x)
    x = self.r0_1(x)
    x = self.r0_2(x)
    x = self.r0_3(x)
    x = self.r0_4(x)
    x = self.r0_5(x)
        
    return x        

class HighUp(layers.Layer):
  def __init__(self,filters):
    super(HighUp,self).__init__()

    self.r0_0 = layers.Conv2D(kernel_size = 1,strides = 1,filters = filters,padding = 'same')
    self.r0_1 = layers.BatchNormalization()
    self.r0_2 = layers.Activation('relu') 

    
  def call(self,inputs):
    x = inputs   
      
    x = self.r0_0(x)
    x = self.r0_1(x)
    x = self.r0_2(x)
        
    return x        


class DecoderTransposeX2Block(layers.Layer):
    def __init__(self,filters,use_batchnorm=False,out = 'relu'):
        super(DecoderTransposeX2Block,self).__init__()

        if filters > 64:
          self.l0 = HighDecoderTransposeX2Block(filters) 
        else:
          self.l0 = LowDecoderTransposeX2Block(filters) 
  
    def call(self,inputs):
        x = inputs
          
        x = self.l0(x)

        return x


class ResBlock(layers.Layer):
    def __init__(self,filters):
        super(ResBlock,self).__init__()
        
        if filters > 64:
          self.l0 = HighResBlock(filters) 
        else:
          self.l0 = LowResBlock(filters) 
  
    def call(self,inputs):
        x = inputs
          
        x = self.l0(x)

        return x
         

class SkelaSmooth(layers.Layer):
    def __init__(self,filters):
        super(SkelaSmooth,self).__init__()
        
        if filters > 64:
          self.l0 = HighSkelaSmooth(filters) 
        else:
          self.l0 = LowSkelaSmooth(filters) 
  
    def call(self,inputs):
        x = inputs
          
        x = self.l0(x)

        return x
        

class LightSmooth(layers.Layer):
    def __init__(self,filters):
        super(LightSmooth,self).__init__()
         
        if filters > 64:
          self.l0 = HighLightSmooth(filters) 
        else:
          self.l0 = LowLightSmooth(filters) 
  
    def call(self,inputs):
        x = inputs
          
        x = self.l0(x)

        return x

class Smooth(layers.Layer):
    def __init__(self,filters):
        super(Smooth,self).__init__()
        
        if filters > 64:
          self.l0 = HighSmooth(filters) 
        else:
          self.l0 = LowSmooth(filters) 
  
    def call(self,inputs):
        x = inputs
          
        x = self.l0(x)

        return x

class Down(layers.Layer):
    def __init__(self,filters):
        super(Down,self).__init__()
        
        if filters > 64:
          self.l0 = HighDown(filters) 
        else:
          self.l0 = LowDown(filters) 
  
    def call(self,inputs):
        x = inputs
          
        x = self.l0(x)

        return x

class Up(layers.Layer):
    def __init__(self,filters):
        super(Up,self).__init__()
        
        if filters > 64:
          self.l0 = HighUp(filters) 
        else:
          self.l0 = LowUp(filters) 
  
    def call(self,inputs):
        x = inputs
          
        x = self.l0(x)

        return x


class Infer(layers.Layer):
  def __init__(self,filters):
    super(Infer,self).__init__()
    
    self.l0 = DecoderTransposeX2Block(filters,
            use_batchnorm = True,out = 'relu')
    self.l1 = LightSmooth(filters)
      
  def call(self,inputs):
    x = inputs

    x = self.l0(x)  
    x = self.l1(x)  
     
    return x

class FilterExpand(layers.Layer):
  def __init__(self,filters):
    super(FilterExpand,self).__init__()
    
    self.l0 = Up(filters//4)
    self.l1 = LightSmooth(filters//4)
    self.l2 = Up(filters//2)
    self.l3 = LightSmooth(filters//2)
    self.l4 = Smooth(filters)
 
  def call(self,inputs):
    x = inputs
 
    x = self.l0(x)  
    x = self.l1(x)  
    x = self.l2(x)  
    x = self.l3(x)  
    x = self.l4(x)  
    
    return x

  
class FilterCrush(layers.Layer):
  def __init__(self,filters):
    super(FilterCrush,self).__init__()
        
    self.l0_0 = Down(filters//2)
    self.l0_1 = LightSmooth(filters//2)
    
    self.lm1_0 = Down(filters//4)
    self.lm1_1 = LightSmooth(filters//4)
    
    self.lv1_0 = Down(filters//4)
    self.lv1_1 = LightSmooth(filters//4)

    self.lm2 = Conv2D(1,3,padding = 'same')    
    self.lv2 = Conv2D(1,3,padding = 'same')    
     
  def call(self,inputs):
    x = inputs
  
    x = self.l0_0(x)  
    x = self.l0_1(x)  
    
    mean = self.lm1_0(x)  
    mean = self.lm1_1(mean)  
    mean = self.lm2(mean)  
    
    logvar = self.lv1_0(x)  
    logvar = self.lv1_1(logvar)  
    logvar = self.lv2(logvar)  

    return mean,logvar  

class Visual(layers.Layer):
  def __init__(self,dim = 1,trainable = True):
    super(Visual,self).__init__(trainable = trainable)
 
    self.v0 = Down(8)
    self.v1 = LightSmooth(8)
     
  def call(self,inputs):
    x = inputs
         
    v = self.v0(x)
    v = self.v1(v)
    
    return v


class DecoderBlock(layers.Layer):
  def __init__(self,in_filters,out_filters):
    super(DecoderBlock,self).__init__()

    self.l0 = DecoderTransposeX2Block(in_filters,
            use_batchnorm = True,out = 'relu')
    self.l1 = SkelaSmooth(in_filters)
    self.l2 = Down(out_filters)
    self.l3 = LightSmooth(out_filters)
    
  def call(self,inputs):
    s2,s1 = inputs

    s2 = self.l0(s2)  
    s2 = self.l1(s2)  

    s1 = tf.concat([s1,s2],axis = -1)
    
    s1 = self.l2(s1)
    s1 = self.l3(s1)

    return s1 

class DecoderBlockNoConcat(layers.Layer):
  def __init__(self,in_filters,out_filters):
    super(DecoderBlockNoConcat,self).__init__()

    self.l0 = DecoderTransposeX2Block(in_filters,
            use_batchnorm = True,out = 'relu')
    self.l1 = SkelaSmooth(in_filters)
    self.l2 = Down(out_filters)
    self.l3 = LightSmooth(out_filters)
    
  def call(self,inputs):
    s2 = inputs

    s2 = self.l0(s2)  
    s2 = self.l1(s2)  
 
    s2 = self.l2(s2)
    s2 = self.l3(s2)

    return s2
    

class Decoder(layers.Layer):
  def __init__(self,trainable = True):
    super(Decoder,self).__init__(trainable=trainable)
    
    self.d4 = DecoderBlock(512,256)
    self.d3 = DecoderBlock(256,128)
    self.d2 = DecoderBlock(128,64)
    self.d1 = DecoderBlock(64,32)
    self.d0 = DecoderBlockNoConcat(32,16)
  
  def call(self,inputs):
    if len(inputs) == 2:
      z,k = inputs
    
      inputs = (tf.concat([z[0],k[0]],axis = -1),
                tf.concat([z[1],k[1]],axis = -1),
                tf.concat([z[2],k[2]],axis = -1),
                tf.concat([z[3],k[3]],axis = -1),
                tf.concat([z[4],k[4]],axis = -1))
    
    s5,s4,s3,s2,s1 = inputs
 
    d4 = self.d4((s5,s4))
    d3 = self.d3((d4,s3))
    d2 = self.d2((d3,s2))
    d1 = self.d1((d2,s1))
    d0 = self.d0(d1)

    return d0

def get_backbone():
  model_bb,_ = Classifiers.get(name = 'mobilenetv2')
  model_bb = model_bb(input_shape = (SHAPE,SHAPE,RGB),include_top = False,weights = None) 
        
  all_layers = model_bb.layers
  i = 1
  l1_inputs = layers.Input((SHAPE,SHAPE,RGB))
  l1 = tf.keras.applications.mobilenet_v2.preprocess_input(l1_inputs)
  while all_layers[i].name  != 'block_1_pad':
    l1 = all_layers[i](l1)
    i += 1
  l1 = Down(32)(l1)
  l1 = Model(inputs=l1_inputs,outputs = l1)
          

  add = []
  l2_inputs = layers.Input((SHAPE//2,SHAPE//2,32))
  l2 = Up(96)(l2_inputs)
  while all_layers[i].name  != 'block_2_add':
    l2 = all_layers[i](l2)
    if (all_layers[i].name == 'block_1_project_BN' or
      all_layers[i].name == 'block_2_project_BN'):
      add.append(l2)
    i += 1

  l2 = all_layers[i](add)
  i += 1

  while all_layers[i].name  != 'block_3_pad':
    l2 = all_layers[i](l2)
    i += 1

  l2 = Down(64)(l2)
  l2 = Model(inputs=l2_inputs,outputs = l2)

  add = []
  l3_inputs = layers.Input((SHAPE//4,SHAPE//4,64))
  l3 = Up(144)(l3_inputs)
  while all_layers[i].name  != 'block_4_add':
    l3 = all_layers[i](l3)
    if (all_layers[i].name == 'block_3_project_BN' or
      all_layers[i].name == 'block_4_project_BN'):
      add.append(l3)
    i += 1

  l3 = all_layers[i](add)
  i += 1

  add = []
  while all_layers[i].name  != 'block_5_add':
    l3 = all_layers[i](l3)
    if (all_layers[i].name == 'block_4_add' or
      all_layers[i].name == 'block_5_project_BN'):
      add.append(l3)
    i += 1

  l3 = all_layers[i](add)
  i += 1

  while all_layers[i].name  != 'block_6_pad':
    l3 = all_layers[i](l3)
    i += 1
  
  l3 = Down(128)(l3)
  l3 = Model(inputs=l3_inputs,outputs = l3)

  add = []
  l4_inputs = layers.Input((SHAPE//8,SHAPE//8,128))
  l4 = Up(192)(l4_inputs)
  while all_layers[i].name  != 'block_7_add':
    l4 = all_layers[i](l4)
    if (all_layers[i].name == 'block_6_project_BN' or
      all_layers[i].name == 'block_7_project_BN'):
      add.append(l4)
    i += 1

  l4 = all_layers[i](add)
  i += 1

  add = []
  while all_layers[i].name  != 'block_8_add':
    l4 = all_layers[i](l4)
    if (all_layers[i].name == 'block_7_add' or
      all_layers[i].name == 'block_8_project_BN'):
      add.append(l4)
    i += 1

  l4 = all_layers[i](add)
  i += 1

  add = []
  while all_layers[i].name  != 'block_9_add':
    l4 = all_layers[i](l4)
    if (all_layers[i].name == 'block_8_add' or
      all_layers[i].name == 'block_9_project_BN'):
      add.append(l4)
    i += 1

  l4 = all_layers[i](add)
  i += 1

  while all_layers[i].name  != 'block_11_depthwise':
    l4 = all_layers[i](l4)
    i += 1
  l4 = Down(256)(l4)
  l4 = Model(inputs=l4_inputs,outputs = l4)

  add = []
  l5_inputs = layers.Input((SHAPE//16,SHAPE//16,256))
  l5 = Up(576)(l5_inputs)
  while all_layers[i].name  != 'block_11_add':
    l5 = all_layers[i](l5)
    if (all_layers[i].name == 'block_10_project_BN' or
      all_layers[i].name == 'block_11_project_BN'):
      add.append(l5)
    i += 1

  l5 = all_layers[i](add)
  i += 1

  add = []
  while all_layers[i].name  != 'block_12_add':
    l5 = all_layers[i](l5)
    if (all_layers[i].name == 'block_11_add' or
      all_layers[i].name == 'block_12_project_BN'):
      add.append(l5)
    i += 1

  l5 = all_layers[i](add)
  i += 1

  add = []
  while all_layers[i].name  != 'block_14_add':
    l5 = all_layers[i](l5)
    if (all_layers[i].name == 'block_13_project_BN' or
      all_layers[i].name == 'block_14_project_BN'):
      add.append(l5)
    i += 1

  l5 = all_layers[i](add)
  i += 1

  add = []
  while all_layers[i].name  != 'block_15_add':
    l5 = all_layers[i](l5)
    if (all_layers[i].name == 'block_14_add' or
      all_layers[i].name == 'block_15_project_BN'):
      add.append(l5)
    i += 1

  l5 = all_layers[i](add)
  i += 1

  while all_layers[i].name  != 'out_relu':
    l5 = all_layers[i](l5)
    i += 1
  l5 = all_layers[i](l5)
  l5 = Down(512)(l5)
  l5 = Model(inputs=l5_inputs,outputs = l5)

  return l1,l2,l3,l4,l5
	
