from scipy.special import gamma
import tensorflow as tf
import numpy as np

def bandwith(dim_v):
    # Assuming here that after the mapping the latent space has some sort of gaussian distribution.
    gz = 2 * gamma(0.5 * (dim_v+1)) / gamma(0.5 * dim_v)
    return 1. / (2. * gz)

def rbf_gauss(u, v, gamma_):
    dist_table_matrix = tf.expand_dims(u, 0) - tf.expand_dims(v, 1)
    l2_dist = tf.reduce_sum(dist_table_matrix**2, axis=-1)
    rbf = tf.exp(-gamma_*l2_dist)
    return rbf
    
def HSIC(u, v, gamma=None):
    dim_u = u.shape.as_list()[1]
    dim_v = v.shape.as_list()[1]
    
    if gamma is None:
        gamma_u = bandwith(dim_u)
        gamma_v = bandwith(dim_v)
    else: 
        gamma_u = gamma
        gamma_v = gamma
        
    uu = rbf_gauss(u, u, gamma_=gamma_u)
    vv = rbf_gauss(v, v, gamma_=gamma_v)
    
    # HSIC = E_xx'yy'[k(x,x')l(y,y')] + E_xx'[k(x,x')]E_yy'[l(y,y')] - 2 E_xy[ E_x'[k(x,x')] E_y'[l(y,y')] ]
    term_1 = tf.reduce_mean(uu * vv)
    term_2 = tf.reduce_mean(uu) * tf.reduce_mean(vv)
    term_3 = 2 * tf.reduce_mean( tf.reduce_mean(uu, axis=1) * tf.reduce_mean(vv, axis=1) )
    value = tf.sqrt(term_1 + term_2 - term_3)
    return value