"""
Created on Sep 30 21:30:45 2019

MT1D modeling using analytic method

@author: lian_liu
"""
# In[]:
#0 import modules
import tensorflow as tf
import numpy as np
import cmath as cm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[]:
#1 define mt1d forward function -- tensor
def mt1d_analytic_tensor(resis):
    resis = 10.0 ** (resis * 4.0)
    sig = 1.0 / resis
    
    #1.1 get the number of the model and layers
    nm = tf.shape(sig)[0]
    nz = tf.shape(sig)[1]
    
    #1.2 define the constants
    freq = tf.reshape(10.0 ** tf.linspace(4.0, 0, 25), [-1, 1])
    
    pi = tf.constant(cm.pi)
    mu  = tf.constant(4.0e-7) * pi
    # dz = tf.reshape(5.0 * (1.23 ** tf.range(0.0, nz - 1, 1)), [1,-1])
    dz = tf.reshape(20. + (10.** (0.115 * tf.range(1., nz, 1))), [1, -1])
    dz = tf.concat([tf.constant([[20.]]), dz], 1)
    II  = tf.constant(1.0j, dtype = tf.dtypes.complex64)
    
    #1.3 initialize the returning arguments
    nf  = tf.shape(freq)[0]
    rho = tf.zeros([nf, nm], dtype = tf.dtypes.float32)
    # phs = tf.zeros([nf, nm], dtype = tf.dtypes.float32)
    zxy = tf.zeros([nf, nm], dtype = tf.dtypes.complex64)
    Z   = tf.zeros([nf, nm], dtype = tf.dtypes.complex64)
    
    #1.4 calculate Z_bottom (Z_n) of all ferquency and model
    omega = 2.0 * pi * freq # omega = 2.0 * np.pi * np.repeat(freq, rho.shape[0], axis=0)
    Z = tf.sqrt(- II * tf.cast(omega * mu / sig[:, nz-1], tf.complex64))
    
    #1.5 loop Z
    n = tf.constant(-1)
    m = nz - 2
    def cond(m, n, Z):
        return m > n
    
    def Z_recursion(m, n, Z):
        km = tf.sqrt(- II * tf.cast(omega * mu * sig[:, m], tf.complex64))
        Z0 = -II * tf.cast(omega * mu, tf.complex64) / km
        Z = tf.math.exp(- 2.0 * km * tf.cast(dz[:, m], tf.complex64)) * (Z - Z0) / (Z + Z0)
        Z = Z0 * (1 + Z) / (1 - Z)
        m = m - 1
        return m, n, Z
    m, n, Z = tf.while_loop(cond, Z_recursion, [m, n, Z])
    
    zxy = Z
    
    rho = tf.math.abs(zxy) * tf.math.abs(zxy) / (omega * mu)
    
    rho = tf.math.log(rho) / tf.math.log(10.0) / 4.0
    
    # phs = - tf.math.atan2(tf.math.imag(zxy), tf.math.real(zxy)) * 180.0 / pi
    
    return tf.transpose(rho)
    # return rho, phs, zxy
    
# In[]:
#2 define mt1d forward function -- numpy    
def mt1d_analytic(freq, dz, sig, errorfloor=0.):
    #2.1 define the constants
    mu  = 4.0e-7 * np.pi
    II  = cm.sqrt(-1)

    #2.2 get the number of the model and layers
    nm, nz = sig.shape

    #2.3 initialize the returning arguments
    nf  = len(freq)
    rho = np.zeros((nf, nm), dtype = np.float32)
    phs = np.zeros((nf, nm), dtype = np.float32)
    zxy = np.zeros((nf, nm), dtype = np.complex64)
    Z   = np.zeros((nf, nm), dtype = np.complex64)

    #2.4 calculate Z_bottom (Z_n) of all ferquency and model
    omega = 2.0 * np.pi * freq #omega = 2.0 * np.pi * np.repeat(freq, rho.shape[0], axis=0)
    Z     = np.sqrt(- II * omega * mu / sig[:, nz-1])
    
    #2.5 loop Z
    for m in range(nz - 2, -1, -1):
        km = np.sqrt(-II * omega * mu * sig[:, m])
#        print(km)
#        print(km.shape)
        Z0 = -II * omega * mu / km
        Z  = np.exp(-2.0 * km * dz[:, m]) * (Z - Z0) / (Z + Z0)
        Z  = Z0 * (1 + Z) / (1 - Z)
    zxy = Z
    z_sd = abs(zxy) * errorfloor
    z_sd_r = np.random.normal(0., 1., [nf, 1]) * z_sd
    z_sd_i = np.random.normal(0., 1., [nf, 1]) * z_sd
    zxy = zxy + z_sd_r + II * z_sd_i
    
    
    rho = abs(zxy * zxy) / (omega * mu)
    np.random.seed(0)
    rho_sd = 0.4 * abs(zxy) * abs(z_sd + II * z_sd) / (freq * 4.e-4 * np.pi * 4.e-4 * np.pi)
    # rho_sd = 0.4 * abs(zxy) * abs(z_sd + II * z_sd_i) / (freq * mu)
    
    # phs = - np.arctan2(zxy.imag, zxy.real) * 180.0 / np.pi
    
    return zxy, z_sd, rho, rho_sd
    # return rho
