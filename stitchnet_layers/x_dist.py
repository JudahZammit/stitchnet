from tensorflow.keras import layers as tfkl
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions
from stitchnet_layers.state import State
l = State.layers
from shared_files.param import *

def get_full_layer_key(func,key):
    """ get the full func name for proper indexing
        into the state dictionary """  

    return func.__name__ + '_' + key  


def visual_to_x_dist_param(visual,layer_key):
    """ Return the parameter to one to either alpha or beta of the beta distribution.
        Allowed values for layer_key is one of: alpha or beta """
    
    full_layer_key = get_full_layer_key(visual_to_x_dist_param,layer_key) 
    
    raw_output = l[full_layer_key](visual)
    clipped_output = tfkl.Activation('relu')(raw_output) + 1e-7
    
    return clipped_output

def x_dist_param_to_dist(alpha,beta):
    """ Take the parametres to x distribution and return the 
        full tfp distribution """
    
    return tfpd.Beta(alpha,beta)


def visual_to_x_dist(alpha_visual,beta_visual):
    """ Take two visuals, alpha_visual and beta_visual, and return the full x dist"""
  
    alpha = visual_to_x_dist_param(alpha_visual,'alpha')
    beta = visual_to_x_dist_param(beta_visual,'beta')

    dist = x_dist_param_to_dist(alpha,beta)

    return dist    

def visual_to_x_dist(alpha_visual,beta_visual,lab):
    """ Take two visuals, alpha_visual and beta_visual, and return the full x dist"""

    alpha = visual_to_x_dist_param(alpha_visual,lab+'_alpha')
    beta = visual_to_x_dist_param(beta_visual,lab+'_beta')

    dist = x_dist_param_to_dist(alpha,beta)

    return dist
    
def clip_x(x):
    """ Modify x to work with our chosen distribution"""
    x = x/255
  
    rep = tf.ones_like(x)*(1-1e-7)      
    x = tf.where(x>rep,rep,x)    

    rep = tf.ones_like(x)*(1e-7)      
    x = tf.where(x<rep,rep,x)    

    return x

def x_ll(x,dist):
    """ Find the likelihood of an x under the distribution dist"""

    x_clipped = clip_x(x)
  
    likelihood = dist.prob(x_clipped)

    sh = tf.shape(x)[0]
    sh = tf.cast(sh,'float32')
    #ll = tf.reduce_mean(tf.math.log(likelihood + 1e-7),axis = [1,2,3])
    ll = tf.math.log(likelihood + 1e-7)
    ll_scaled = sh*ll/(BS*BUCKETS)    

    return ll_scaled


def dist_to_x(dist):
    """ get a sample, or some other representative, from dist"""

    return dist.mean()
