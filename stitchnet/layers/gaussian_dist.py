import tensorflow as tf
import math
import tensorflow.keras.layers as tfkl
import tensorflow.keras.backend as K
from shared_files.param import SHAPE,BS
from stichnet.layers.state import State
l = State.layers

def safe_output(x):
    """ Avoid NaN when KL is zero"""

    if l['KL'] == 0:
        return 0.0
    else:
        return x


def scale_density(density):
    
    summed = K.sum(density,axis = -1)
    scaled = summed/(SHAPE*SHAPE)

    return scaled

def gaussian_ll(x,mu,log_var):
    
    x = tfkl.Flatten()(x)
    mu = tfkl.Flatten()(mu)
    log_var = tfkl.Flatten()(log_var)

    constant = -.5 * K.log(2*math.pi)
    density1 = l['KL']*constant 
    density2 = - l['KL']*log_var/2 
    density3 = - l['KL']*((x - mu)/(2*K.exp(log_var) + 1e-8))*(x - mu)
    density = density1 + density2 + density3

    scaled = scale_density(density)
    
    safe = safe_output(scaled)

    return safe

def gaussian_sample(mu,log_var):
    
    epsilon = K.random_normal(shape = K.shape(mu))
    sample = mu + K.exp(0.5 * log_var) * epsilon
    
    return sample

def get_unit_gaussian_dist():
    
    mu = l['zeros']
    log_var = l['zeros']

    return mu,log_var
         
