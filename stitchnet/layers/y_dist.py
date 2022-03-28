from tensorflow.keras import layers as tfkl
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions
from stichnet.layers.state import State
l = State.layers
from shared_files.param import *

def visual_to_y_dist_param(visual):
    
    full_layer_key = 'visual_to_y_dist_param'
    
    raw_output = l[full_layer_key](visual)
    clipped_output = tfkl.Activation('softmax')(raw_output)
    
    return clipped_output


def y_ll(y,dist):
    """ Find the likelihood of an y under the distribution dist"""
 
    ll = tf.keras.losses.CategoricalCrossentropy()(y,dist)
    tf.print(ll)

    sh = tf.shape(y)[0]
    sh = tf.cast(sh,'float32')
    ll_scaled = sh*ll/BS    

    return ll_scaled
