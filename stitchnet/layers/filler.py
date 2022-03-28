from stichnet.layers.gaussian_dist import * 
from stichnet.layers.x_dist import *
from stichnet.layers.y_dist import *
from stichnet.layers.state import State
from shared_files.param import *
l = State.layers
import tensorflow.keras as tfk
import numpy as np
from focal_loss import sparse_categorical_focal_loss

def get_qhat_znt_expand_direct(zn_1_expand,n):
  
  zn_expand = l['z{}_expand_to_z{}_expand'.format(n-1,n)](zn_1_expand)

  return zn_expand


def get_qhat_zt_expand_direct(x):
    
  qhat_zt_expand_direct = {}
   
  zn_1t_expand = x
  for n in range(1,6):
      zn_1t_expand = get_qhat_znt_expand_direct(zn_1t_expand,n)
      qhat_zt_expand_direct['qhat_zt_expand_direct{}'.format(n)] = zn_1t_expand

  return qhat_zt_expand_direct

def get_all_qhat_z(all_expands):
  
  qhat_z = []

  for i in range(len(all_expands)):
    qhat_z_layer = l['z{}_expand_to_z{}'.format(5-i,5-i)](all_expands['qhat_zt_expand_direct{}'.format(5-i)])
    qhat_z.append(qhat_z_layer)

  return qhat_z

def get_qhat_z(x):
  
  all_expands = get_qhat_zt_expand_direct(x)
 
  qhat_z = get_all_qhat_z(all_expands) 
    
  return qhat_z


def combine_params(q_mean,q_logvar,p_mean,p_logvar):
  
  q_var = K.exp(q_logvar)
  q_var_inv = 1/q_var

  p_var = K.exp(p_logvar)
  p_var_inv = 1/p_var

  var = 1/(p_var_inv + q_var_inv)
  logvar = K.log(var)

  mean_numerator = q_mean*q_var_inv + p_mean*p_var_inv
  mean_denominator = (p_var_inv + q_var_inv)
        
  mean = mean_numerator/mean_denominator

  return mean,logvar

def get_level_info(qhat_zn_1,zn_expanded,zn_sample,level,gen):
    
  if level == 5:
    p_zn_1 = get_unit_gaussian_dist()
  else:
    zn_1_expanded = l['z{}_expand_to_z{}_expand'.format(level+1,level)](zn_expanded)
    p_zn_1 = l['z{}_expand_to_z{}'.format(level,level)](zn_1_expanded)   

  
  if gen:
    q_zn_1 = p_zn_1
  else:
    q_zn_1 = combine_params(qhat_zn_1[0],qhat_zn_1[1],
                            p_zn_1[0],p_zn_1[1])
    
  zn_1_sample = gaussian_sample(q_zn_1[0],q_zn_1[1])
  zn_1_expanded = l['z{}_to_z{}_expand'.format(level,level)](zn_1_sample)    
   

  return p_zn_1,q_zn_1,zn_1_expanded,zn_1_sample


def z_information(qhat_z,gen = False):
  
  p_z = []
  q_z = []
  z_expanded = []
  z_sample = []

  level_5 = get_level_info(qhat_z[0],None,None,5-0,gen)
  p_z.append(level_5[0])
  q_z.append(level_5[1])
  z_expanded.append(level_5[2])
  z_sample.append(level_5[3])
  for i in range(1,5):
    level = 5 - i
    level_n_1 = get_level_info(qhat_z[i],z_expanded[-1],z_sample[-1],level,gen)
    p_z.append(level_n_1[0])
    q_z.append(level_n_1[1])
    z_expanded.append(level_n_1[2])
    z_sample.append(level_n_1[3])

  out={}
  out['p_z'] = p_z
  out['q_z'] = q_z
  out['z_expanded'] = z_expanded
  out['z_sample'] = z_sample

  return out


def get_decoded_z(z_expanded):
    return l['decoder'](z_expanded[-1])

def get_visuals(decoded_z,lab):

    alpha_visual = l[lab+'_alpha_visual'](decoded_z)
    beta_visual = l[lab+'_beta_visual'](decoded_z)

    return alpha_visual,beta_visual

def get_visual(decoded_z,lab):

    return l[lab+'_visual'](decoded_z)

def create_output_dict(z_sample,x_reconstructed,x_blk_reconstructed,x_hel_reconstructed,x_con_reconstructed,x_ggo_reconstructed,p_y):
  
  out = {}
  out['z5_sample'] = z_sample[0]
  out['z4_sample'] = z_sample[1]
  out['z3_sample'] = z_sample[2]
  out['z2_sample'] = z_sample[3]
  out['z1_sample'] = z_sample[4]
  out['x_reconstructed'] = x_reconstructed
  out['x_ggo_reconstructed'] = x_ggo_reconstructed
  out['x_con_reconstructed'] = x_con_reconstructed
  out['x_blk_reconstructed'] = x_blk_reconstructed
  out['x_hel_reconstructed'] = x_hel_reconstructed
  out['y_reconstructed'] = p_y

  return out

def create_loss_dict(xent,cce,z_sample,p_z,q_z):
  
  # get losses
  loss_dict = {}

  # x recon loss
  loss_dict['XENT'] = xent  
  loss_dict['CCE'] = cce

  # p_z loss 
  for i in range(5):
    loss_dict['p_z{}'.format(5-i)] = -gaussian_ll(z_sample[i],p_z[i][0],p_z[i][1]) 
  
  # q_z loss 
  for i in range(5):
    loss_dict['q_z{}'.format(5-i)] = gaussian_ll(z_sample[i],q_z[i][0],q_z[i][1])
  
  loss = 0
  for x in loss_dict.values():
    loss += x
  
  loss_dict['loss'] = loss
  loss_dict['KL'] = loss - loss_dict['XENT'] - loss_dict['CCE']

  return loss_dict

def predict(inputs,Y_hel,Y_lab,gen):
  x = inputs 
 
  qhat_z = get_qhat_z(x)
  
  z_info = z_information(qhat_z,gen = gen)
   
  all_out = {}
  all_loss = {'KL':0,'loss':0,'XENT':0,'CCE':0}

  z_expand = z_info['z_expanded']   
 
  all_decoded = get_decoded_z(z_expand)

  hel_alpha_visual,hel_beta_visual = get_visuals(all_decoded,'hel')
  con_alpha_visual,con_beta_visual = get_visuals(all_decoded,'con')
  ggo_alpha_visual,ggo_beta_visual = get_visuals(all_decoded,'ggo')
  blk_alpha_visual,blk_beta_visual = get_visuals(all_decoded,'blk')
  y_visual = get_visual(all_decoded,'y')

  p_x_hel = visual_to_x_dist(hel_alpha_visual,hel_beta_visual,'hel')
  p_x_con = visual_to_x_dist(con_alpha_visual,con_beta_visual,'con')
  p_x_ggo = visual_to_x_dist(ggo_alpha_visual,ggo_beta_visual,'ggo')
  p_x_blk = visual_to_x_dist(blk_alpha_visual,blk_beta_visual,'blk')
  p_y = visual_to_y_dist_param(y_visual)

  #hel_cce = tf.keras.losses.CategoricalCrossentropy(reduction = tf.keras.losses.Reduction.NONE)(Y_hel,p_y[:BS//3])
  #lab_cce = tf.keras.losses.CategoricalCrossentropy(reduction = tf.keras.losses.Reduction.NONE)(Y_lab,p_y[-BS//3:])
  cce = sparse_categorical_focal_loss(tf.concat([tf.math.argmax(Y_hel,axis = -1),tf.math.argmax(Y_lab,axis = -1)],axis = 0),
                                      tf.concat([p_y[:BS//3],p_y[2*BS//3:]],axis = 0),
                                      gamma = 2)
  cce = tf.reduce_mean(cce,axis = [1,2])
  #hel_cce = tf.reduce_mean(hel_cce,axis = [1,2])
  #lab_cce = tf.reduce_mean(lab_cce,axis = [1,2])
  cce = tf.concat([cce[:BS//3],cce[:BS//3]*0,cce[BS//3:]],axis = 0) * 100

  x_blk_reconstructed = dist_to_x(p_x_blk)[...,0]
  x_hel_reconstructed = dist_to_x(p_x_hel)[...,0]
  x_con_reconstructed = dist_to_x(p_x_con)[...,0]
  x_ggo_reconstructed = dist_to_x(p_x_ggo)[...,0]
  x_reconstructed = ((x_blk_reconstructed*p_y[...,0]) 
                     + (x_hel_reconstructed*p_y[...,1]) 
                     + (x_con_reconstructed*p_y[...,2]) 
                     + (x_ggo_reconstructed*p_y[...,3]))

  blk_xent = -x_ll(x,p_x_blk)[...,0]
  hel_xent = -x_ll(x,p_x_hel)[...,0]
  ggo_xent = -x_ll(x,p_x_ggo)[...,0]
  con_xent = -x_ll(x,p_x_con)[...,0]
  xent = ( (blk_xent*p_y[...,0]) 
         + (hel_xent*p_y[...,1]) 
         + (con_xent*p_y[...,2]) 
         + (ggo_xent*p_y[...,3]))
  xent = tf.reduce_mean(xent,axis=[1,2])

  out = create_output_dict(z_info['z_sample'],
                           x_reconstructed,
                           x_blk_reconstructed,
                           x_hel_reconstructed,
                           x_con_reconstructed,
                           x_ggo_reconstructed,
                           p_y)
  all_out.update(out)
   
  loss_dict = create_loss_dict(xent,cce,z_info['z_sample'],
                                z_info['p_z'],z_info['q_z'])
  for key in all_loss:
      all_loss[key] += loss_dict[key] 

  return all_out,all_loss
 
class myModel(tfk.Model):
    def __init__(self):
        super(myModel,self).__init__()
        
        self.l = State.layers

    def call(self,inputs,gen = False):
        
        T_hel,Y_hel,T_ncp,T_lab,Y_lab = inputs
        X = tf.concat([T_hel,T_ncp,T_lab],axis = 0)
        out,loss_dict = predict(X,Y_hel,Y_lab,gen = gen)
        out['loss'] = loss_dict['loss'] 
        out['Actual KL'] = loss_dict['KL']/l['KL'] 
        out['XENT'] = loss_dict['XENT'] 
        out['CCE'] = loss_dict['CCE'] 
        self.add_loss(loss_dict['loss'])
        
        Y_lab_pred = out['y_reconstructed'][2*BS//3:]
        Y_lab_pred = tf.one_hot(tf.math.argmax(Y_lab_pred,axis = -1),4)
        Y_lab = tf.one_hot(tf.math.argmax(Y_lab,axis = -1),4)

        con_prec = tf.math.reduce_sum(Y_lab_pred[...,3] * Y_lab[...,3],axis = [1,2]) / (tf.math.reduce_sum(Y_lab_pred[...,3],axis = [1,2]) + 1e-7)
        con_recall =tf.math.reduce_sum(Y_lab_pred[...,3] * Y_lab[...,3],axis = [1,2]) /  (tf.math.reduce_sum(Y_lab[...,3],axis = [1,2]) + 1e-7)
        con_f1 = (2*con_prec * con_recall)/(con_prec + con_recall + 1e-7)
        ggo_prec = tf.math.reduce_sum(Y_lab_pred[...,2] * Y_lab[...,2],axis = [1,2]) / (tf.math.reduce_sum(Y_lab_pred[...,2],axis = [1,2]) + 1e-7)
        ggo_recall =tf.math.reduce_sum(Y_lab_pred[...,2] * Y_lab[...,2],axis = [1,2]) /  (tf.math.reduce_sum(Y_lab[...,2],axis = [1,2]) + 1e-7)
        ggo_f1 = (2*ggo_prec * ggo_recall)/(ggo_prec + ggo_recall + 1e-7)

        self.add_metric(ggo_f1,name = 'GGO F1',aggregation = 'mean')          
        self.add_metric(con_f1,name = 'CON F1',aggregation = 'mean')          
        self.add_metric(loss_dict['XENT'],name = 'XENT',aggregation = 'mean')          
        self.add_metric(loss_dict['CCE'],name = 'CCE',aggregation = 'mean')          
        self.add_metric(loss_dict['KL']/l['KL'],name = 'Actual KL',aggregation = 'mean')       
        self.add_metric(loss_dict['KL'],name = 'Scaled KL',aggregation = 'mean')       

        return out


