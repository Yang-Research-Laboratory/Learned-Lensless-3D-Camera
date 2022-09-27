# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 22:43:29 2022

@author: fengt
"""

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import keras
# from keras import layers
from keras import backend as K
import keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# %%
# from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
class Hadamard(Layer):
    def __init__(self, **kwargs):
        super(Hadamard, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name = 'kernel',#dtype = 'complex64',
                                      shape = (2,) + input_shape[1:],
                                       # initializer = 'uniform',
                                       # initializer = keras.initializers.RandomUniform(minval = 0., maxval = 1.),
                                       initializer = keras.initializers.lecun_normal(seed = None),
                                       # activation = 'elu',#tf.keras.activations.elu(x, alpha = 1.0),#'relu',
                                      trainable = True)
        super(Hadamard, self).build(input_shape)
    def call(self, x):
        # print(x.shape, self.kernel.shape)
        x = tf.cast(x,tf.complex64)
        x2 = tf.signal.fft2d(x[0,:,:,0])
        # ximag = tf.math.imag(x)
        # xreal = tf.math.real(x)
        # ximag = ximag*self.kernel[0]
        # xreal = xreal*self.kernel[1]
        kernel = tf.complex(self.kernel[0,:,:,0],self.kernel[1,:,:,0])
        kernel = tf.cast(kernel,tf.complex64)
        xhadaf = x2*tf.math.conj(kernel)
        # xhadaf = tf.complex(xreal, ximag)
        xhada = tf.signal.ifft2d(xhadaf)
        xhada = tf.signal.fftshift(xhada)
        xhada = tf.abs(xhada)
        # x[0,:,:,0] = xhada
        xhada2 = tf.expand_dims(xhada,0)
        xhada2 = tf.expand_dims(xhada2,3)
        return xhada2 #x * self.kernel
    def compute_output_shape(self, input_shape):
        # print(input_shape)
        return input_shape
# %% 
def reconMRGB(ds1,ds2,csize):
    """
    reconstruction module for RGB images
    """
    # keras.__version__
    # ds1: input data size rows
    # ds2: input data size colunms
    # csize: cropping size

    hadalayer1 = Hadamard()
    hadalayer2 = Hadamard()
    hadalayer3 = Hadamard()
    conv1layer = layers.Conv2D(1,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.RandomUniform(minval = 0., maxval = 1.),activity_regularizer = regularizers.l2(1e-5))
    rawr = layers.Input(shape = (ds1,ds2,1))
    rawg = layers.Input(shape = (ds1,ds2,1))
    rawb = layers.Input(shape = (ds1,ds2,1))
    hadar = hadalayer1(rawr)
    hadag = hadalayer2(rawg)
    hadab = hadalayer3(rawb)
    hada = layers.concatenate([hadar,hadag,hadab])
    if (csize % 2) ==0:
        hadacrop = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int))))(hada)
    if (csize % 4) ==3:
        hadacrop = layers.Cropping2D(cropping = ((np.rint(688-csize/2).astype(int), np.rint(688-csize/2).astype(int)+1), (np.rint(1032-csize/2).astype(int), np.rint(1032-csize/2).astype(int)+1)))(hada)
    if (csize % 4) ==1:
        hadacrop = layers.Cropping2D(cropping = ((np.rint(688-csize/2).astype(int), np.rint(688-csize/2).astype(int)-1), (np.rint(1032-csize/2).astype(int), np.rint(1032-csize/2).astype(int)-1)))(hada)
    hadacropbn = layers.BatchNormalization()(hadacrop)
    reconM = keras.models.Model(inputs = [rawr,rawg,rawb], outputs = [hadacrop])
    return reconM
# reconM.summary()
# %% 6 
def fullgeneratorRGB(ds1,ds2,csize,rsize):
    """
    full generator reconM+enhanceM RGBcolor
    """
    # keras.__version__
    # ds1: input data size rows
    # ds2: input data size colunms
    # csize: cropping size
    # rsize: enhancement module input size
    hadalayer1 = Hadamard()
    hadalayer2 = Hadamard()
    hadalayer3 = Hadamard()
    x1r = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x1g = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x1b = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x2r = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x2g = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x2b = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x3r = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x3g = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x3b = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x4r = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x4g = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x4b = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x5r = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x5g = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x5b = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x6r = layers.Conv2D(256,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x6g = layers.Conv2D(256,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x6b = layers.Conv2D(256,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x7r = layers.Conv2D(512,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x7g = layers.Conv2D(512,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x7b = layers.Conv2D(512,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x8r = layers.Conv2D(256,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x8g = layers.Conv2D(256,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x8b = layers.Conv2D(256,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x9r = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x9g = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x9b = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x10r = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x10g = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x10b = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x11r = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x11g = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x11b = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x12r = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x12g = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x12b = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x13r = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x13g = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x13b = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x14r = layers.Conv2D(1,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x14g = layers.Conv2D(1,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x14b = layers.Conv2D(1,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    rawr = layers.Input(shape = (ds1,ds2,1))
    rawg = layers.Input(shape = (ds1,ds2,1))
    rawb = layers.Input(shape = (ds1,ds2,1))
    
    hadar = hadalayer1(rawr)
    hadag = hadalayer2(rawg)
    hadab = hadalayer3(rawb)
    if (csize % 2) ==0:
        reconr = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int))))(hadar)
        recong = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int))))(hadag)
        reconb = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int))))(hadab)
    if (csize % 4) ==1:
        reconr = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)-1), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int)-1)))(hadar)
        recong = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)-1), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int)-1)))(hadag)
        reconb = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)-1), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int)-1)))(hadab)
    if (csize % 4) ==3:
        reconr = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)+1), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int)+1)))(hadar)
        recong = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)+1), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int)+1)))(hadar)
        reconb = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)+1), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int)+1)))(hadar)

    # hadarecon = layers.concatenate([reconr,recong,reconb])
    # reconr = layers.Lambda(rnorm)([reconr,hadarecon])
    # recong = layers.Lambda(rnorm)([recong,hadarecon])
    # reconb = layers.Lambda(rnorm)([reconb,hadarecon])
    # reconr = layers.Lambda(sepa,arguments = {'y':0})(hadacrop)
    # recong = layers.Lambda(sepa,arguments = {'y':1})(hadacrop)
    # reconb = layers.Lambda(sepa,arguments = {'y':2})(hadacrop)
    
    reconrc = layers.Resizing(rsize, rsize, interpolation = "bicubic")(reconr)
    recongc = layers.Resizing(rsize, rsize, interpolation = "bicubic")(recong)
    reconbc = layers.Resizing(rsize, rsize, interpolation = "bicubic")(reconb)
    c1r = x1r(reconrc)
    c2r = x2r(c1r)
    c3r = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_2_1r')(c2r)
    c4r = x3r(c3r)
    c5r = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_3_1r')(c4r)
    c6r = x4r(c5r)
    c7r = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_4_1r')(c6r)
    c8r = x5r(c7r)
    c9r = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_5_1r')(c8r)
    c10r = x6r(c9r)
    c11r = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_6_1r')(c10r)
    c13r = x7r(c11r)
    c14r = layers.Dropout(0.5)(c13r)
    c15r = x8r(c14r)
    c16r = x9r(c15r)
    c17r = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c16r)
    c18r = layers.concatenate([c17r,c10r], axis = 3)
    c19r = x10r(c18r)
    c20r = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c19r)
    c21r = layers.concatenate([c20r,c8r], axis = 3)
    c22r = x11r(c21r)
    c23r = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c22r)
    c24r = layers.concatenate([c23r,c6r], axis = 3)
    c25r = x12r(c24r)
    c26r = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c25r)
    c27r = layers.concatenate([c26r,c4r], axis = 3)
    c28r = x13r(c27r)
    c29r = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c28r)
    c30r = layers.concatenate([c29r,c2r], axis = 3)
    c31r = layers.BatchNormalization()(c30r)
    c32r = x14r(c31r)
    
    c1g = x1g(recongc)
    c2g = x2g(c1g)
    c3g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_2_1g')(c2g)
    c4g = x3g(c3g)
    c5g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_3_1g')(c4g)
    c6g = x4g(c5g)
    c7g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_4_1g')(c6g)
    c8g = x5g(c7g)
    c9g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_5_1g')(c8g)
    c10g = x6g(c9g)
    c11g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_6_1g')(c10g)
    # c12g = layers.Dropout(0.5)(c11g)
    # c13g = x7g(c12g)
    c13g = x7g(c11g)
    c14g = layers.Dropout(0.5)(c13g)
    c15g = x8g(c14g)
    c16g = x9g(c15g)
    c17g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c16g)
    c18g = layers.concatenate([c17g,c10g], axis = 3)
    c19g = x10g(c18g)
    c20g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c19g)
    c21g = layers.concatenate([c20g,c8g], axis = 3)
    c22g = x11g(c21g)
    c23g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c22g)
    c24g = layers.concatenate([c23g,c6g], axis = 3)
    c25g = x12g(c24g)
    c26g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c25g)
    c27g = layers.concatenate([c26g,c4g], axis = 3)
    c28g = x13g(c27g)
    c29g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c28g)
    c30g = layers.concatenate([c29g,c2g], axis = 3)
    c31g = layers.BatchNormalization()(c30g)
    c32g = x14g(c31g)
    
    c1b = x1b(reconbc)
    c2b = x2b(c1b)
    c3b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_2_1b')(c2b)
    c4b = x3b(c3b)
    c5b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_3_1b')(c4b)
    c6b = x4b(c5b)
    c7b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_4_1b')(c6b)
    c8b = x5b(c7b)
    c9b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_5_1b')(c8b)
    c10b = x6b(c9b)
    c11b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_6_1b')(c10b)
    c13b = x7b(c11b)
    c14b = layers.Dropout(0.5)(c13b)
    c15b = x8b(c14b)
    c16b = x9b(c15b)
    c17b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c16b)
    c18b = layers.concatenate([c17b,c10b], axis = 3)
    c19b = x10b(c18b)
    c20b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c19b)
    c21b = layers.concatenate([c20b,c8b], axis = 3)
    c22b = x11b(c21b)
    c23b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c22b)
    c24b = layers.concatenate([c23b,c6b], axis = 3)
    c25b = x12b(c24b)
    c26b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c25b)
    c27b = layers.concatenate([c26b,c4b], axis = 3)
    c28b = x13b(c27b)
    c29b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c28b)
    c30b = layers.concatenate([c29b,c2b], axis = 3)
    c31b = layers.BatchNormalization()(c30b)
    c32b = x14b(c31b)
    
    c32 = layers.concatenate([c32r,c32g,c32b])
    fullgenerator = keras.models.Model(inputs = [rawr,rawg,rawb], outputs = c32)
    # fullgenerator.summary()
    return fullgenerator
# %%
def fullgeneratormono(ds1,ds2,csize,rsize):
    """
    full generator reconM+enhanceM single color
    """
    # keras.__version__
    # ds1: input data size rows
    # ds2: input data size colunms
    # csize: cropping size
    # rsize: enhancement module input size
    
    # recon = layers.Input(shape = (None,None,3))
    hadalayer1 = Hadamard()
    hadalayer2 = Hadamard()
    hadalayer3 = Hadamard()
    x1 = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x2 = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x3 = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x4 = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x5 = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x6 = layers.Conv2D(256,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x7 = layers.Conv2D(512,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x8 = layers.Conv2D(256,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x9 = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x10 = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x11 = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x12 = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x13 = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x14 = layers.Conv2D(1,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    
    rawr = layers.Input(shape = (ds1,ds2,1))
    rawg = layers.Input(shape = (ds1,ds2,1))
    rawb = layers.Input(shape = (ds1,ds2,1))
    hadar = hadalayer1(rawr)
    hadag = hadalayer2(rawg)
    hadab = hadalayer3(rawb)
    
    if (csize % 2) ==0:
        reconr = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int))))(hadar)
        recong = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int))))(hadag)
        reconb = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int))))(hadab)
    if (csize % 4) ==1:
        reconr = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)-1), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int)-1)))(hadar)
        recong = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)-1), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int)-1)))(hadag)
        reconb = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)-1), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int)-1)))(hadab)
    if (csize % 4) ==3:
        reconr = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)+1), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int)+1)))(hadar)
        recong = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)+1), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int)+1)))(hadar)
        reconb = layers.Cropping2D(cropping = ((np.rint(ds1/2-csize/2).astype(int), np.rint(ds1/2-csize/2).astype(int)+1), (np.rint(ds2/2-csize/2).astype(int), np.rint(ds2/2-csize/2).astype(int)+1)))(hadar)
    
    hadarecon = layers.concatenate([reconr,recong,reconb])
    reconr = layers.Lambda(rnorm)([reconr,hadarecon])
    recong = layers.Lambda(rnorm)([recong,hadarecon])
    reconb = layers.Lambda(rnorm)([reconb,hadarecon])
    # reconr = layers.Lambda(sepa,arguments = {'y':0})(hadacrop)
    # recong = layers.Lambda(sepa,arguments = {'y':1})(hadacrop)
    # reconb = layers.Lambda(sepa,arguments = {'y':2})(hadacrop)
    # reconr = layers.BatchNormalization()(reconr)
    # recong = layers.BatchNormalization()(recong)
    # reconb = layers.BatchNormalization()(reconb)
    reconrc = layers.Resizing(rsize, rsize, interpolation = "bicubic")(reconr)
    recongc = layers.Resizing(rsize, rsize, interpolation = "bicubic")(recong)
    reconbc = layers.Resizing(rsize, rsize, interpolation = "bicubic")(reconb)
    c1r = x1(reconrc)
    c2r = x2(c1r)
    c3r = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_2_1r')(c2r)
    c4r = x3(c3r)
    c5r = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_3_1r')(c4r)
    c6r = x4(c5r)
    c7r = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_4_1r')(c6r)
    c8r = x5(c7r)
    c9r = layers.MaxPooling2D(pool_size = (2, 2),padding='same',name='mp_5_1r')(c8r)
    c10r=x6(c9r)
    c11r=layers.MaxPooling2D(pool_size=(2, 2),padding='same',name='mp_6_1r')(c10r)
    c12r=layers.Dropout(0.5)(c11r)
    c13r=x7(c12r)
    c14r=layers.Dropout(0.5)(c13r)
    c15r=x8(c14r)
    c16r=x9(c15r)
    c17r=keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(c16r)
    c18r=layers.concatenate([c17r,c10r], axis = 3)
    c19r=x10(c18r)
    c20r=keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(c19r)
    c21r=layers.concatenate([c20r,c8r], axis = 3)
    c22r=x11(c21r)
    c23r=keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(c22r)
    c24r=layers.concatenate([c23r,c6r], axis = 3)
    c25r=x12(c24r)
    c26r=keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(c25r)
    c27r=layers.concatenate([c26r,c4r], axis = 3)
    c28r=x13(c27r)
    c29r=keras.layers.UpSampling2D(size=(2, 2),interpolation='bilinear')(c28r)
    c30r=layers.concatenate([c29r,c2r], axis = 3)
    c31r=layers.BatchNormalization()(c30r)
    c32r=x14(c31r)
    
    c1g=x1(recongc)
    c2g=x2(c1g)
    c3g=layers.MaxPooling2D(pool_size=(2, 2),padding='same',name='mp_2_1g')(c2g)
    c4g=x3(c3g)
    c5g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_3_1g')(c4g)
    c6g = x4(c5g)
    c7g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_4_1g')(c6g)
    c8g = x5(c7g)
    c9g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_5_1g')(c8g)
    c10g = x6(c9g)
    c11g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_6_1g')(c10g)
    c12g = layers.Dropout(0.5)(c11g)
    c13g = x7(c12g)
    c14g = layers.Dropout(0.5)(c13g)
    c15g = x8(c14g)
    c16g = x9(c15g)
    c17g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c16g)
    c18g = layers.concatenate([c17g,c10g], axis = 3)
    c19g = x10(c18g)
    c20g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c19g)
    c21g = layers.concatenate([c20g,c8g], axis = 3)
    c22g = x11(c21g)
    c23g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c22g)
    c24g = layers.concatenate([c23g,c6g], axis = 3)
    c25g = x12(c24g)
    c26g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c25g)
    c27g = layers.concatenate([c26g,c4g], axis = 3)
    c28g = x13(c27g)
    c29g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c28g)
    c30g = layers.concatenate([c29g,c2g], axis = 3)
    c31g = layers.BatchNormalization()(c30g)
    c32g = x14(c31g)
    
    c1b = x1(reconbc)
    c2b = x2(c1b)
    c3b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_2_1b')(c2b)
    c4b = x3(c3b)
    c5b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_3_1b')(c4b)
    c6b = x4(c5b)
    c7b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_4_1b')(c6b)
    c8b = x5(c7b)
    c9b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_5_1b')(c8b)
    c10b = x6(c9b)
    c11b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_6_1b')(c10b)
    c12b = layers.Dropout(0.5)(c11b)
    c13b = x7(c12b)
    c14b = layers.Dropout(0.5)(c13b)
    c15b = x8(c14b)
    c16b = x9(c15b)
    c17b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c16b)
    c18b = layers.concatenate([c17b,c10b], axis = 3)
    c19b = x10(c18b)
    c20b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c19b)
    c21b = layers.concatenate([c20b,c8b], axis = 3)
    c22b = x11(c21b)
    c23b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c22b)
    c24b = layers.concatenate([c23b,c6b], axis = 3)
    c25b = x12(c24b)
    c26b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c25b)
    c27b = layers.concatenate([c26b,c4b], axis = 3)
    c28b = x13(c27b)
    c29b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c28b)
    c30b = layers.concatenate([c29b,c2b], axis = 3)
    c31b = layers.BatchNormalization()(c30b)
    c32b = x14(c31b)
    
    c32 = layers.concatenate([c32r,c32g,c32b])
    # from keras.models import Model
    # fullgenerator = keras.models.Model(inputs = [rawr,rawg,rawb], outputs = [c32,hadarecon,reconr,recong,reconb,hadar,hadag,hadab])
    fullgenerator = keras.models.Model(inputs = [rawr,rawg,rawb], outputs = c32)
    # fullgenerator.summary()
    return fullgenerator
# %%
def discriminator_v1 (channels,dsc):
    """
    discriminator (version 1) smaller number of parameters
    """
    # channels: number of color channels of input images
    # dsc: input image size after cropping
    discriminator_input = layers.Input(shape = (dsc,dsc,channels))
    # discriminator_input2 = layers.Input(shape = (dsc,dsc,channels))
    x = layers.Conv2D(128,(3,3),padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128,(4,4),strides = 2,padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128,(4,4),strides = 2,padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    # x = layers.Dense(1,activation = 'swish')(x)
    x = layers.Dense(1,activation = 'sigmoid')(x)
    # x = layers.Dense(1,activation = 'tanh')(x)
    # discriminator = keras.models.Model([discriminator_input,discriminator_input2],x)
    discriminator = keras.models.Model(discriminator_input,x)
    # discriminator.summary()
    return discriminator
    # discriminator_optimizer = keras.optimizers.RMSprop(lr = 0.0004,clipvalue = 1.0,decay = 1e-8)
    # discriminator.compile(optimizer = discriminator_optimizer,loss = 'binary_crossentropy')
    # discriminator.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics = 'mae')
    # discriminator.compile(optimizer = 'rmsprop',loss = 'mse')
# %%
def discriminator_v2(channels,dsc,drsize):
    """
    discriminator (version 2)
    """
    # channels: number of color channels of input images 3
    # dsc: input image size after cropping 512
    # drsize: input size after resizing 256
    discriminator_input = layers.Input(shape = (dsc,dsc,channels))
    # discriminator_input2 = layers.Input(shape = (dsc,dsc,channels))
    x = layers.Resizing(drsize, drsize, interpolation = "bicubic")(discriminator_input)
    x = layers.Conv2D(16,(3,3),padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(16,(3,3),padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(32,(3,3),strides = 2,padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(32,(3,3),padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64,(3,3),strides = 2,padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64,(3,3),padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128,(3,3),strides = 2,padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128,(3,3),padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256,(3,3),strides = 2,padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256,(3,3),padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(512,(3,3),strides = 2,padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(512,(3,3),padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(1024,(3,3),strides = 2,padding = 'same',kernel_initializer = keras.initializers.lecun_normal(seed = None))(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.Dense(1024)(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(1,activation = 'swish')(x)
    x = layers.Dense(1,activation = 'sigmoid')(x)
    # x = layers.Dense(1,activation = 'tanh')(x)
    # discriminator = keras.models.Model([discriminator_input,discriminator_input2],x)
    discriminator = keras.models.Model(discriminator_input,x)
    # discriminator.summary()
    return discriminator
    # discriminator_optimizer = keras.optimizers.RMSprop(lr = 0.0004,clipvalue = 1.0,decay = 1e-8)
    # discriminator.compile(optimizer = discriminator_optimizer,loss = 'binary_crossentropy')
    # discriminator.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics = 'mae')
    # discriminator.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics = 'mse')
# %% 18 set transfer weights
# weights = reconM.layers[3].get_weights()
# fullgenerator.layers[3].set_weights(weights)
# weights = reconM.layers[4].get_weights()
# fullgenerator.layers[4].set_weights(weights)
# weights = reconM.layers[5].get_weights()
# fullgenerator.layers[5].set_weights(weights)
# %% 19 layer trainable
# fullgenerator.layers[3].trainable = False
# fullgenerator.layers[4].trainable = False
# fullgenerator.layers[5].trainable = False
# # generator.layers[8].trainable = False
# %% 20 layer trainable
# fullgenerator.layers[3].trainable = True
# fullgenerator.layers[4].trainable = True
# fullgenerator.layers[5].trainable = True
# # generator.layers[8].trainable = True
# %% 29-2
# activation = feature_extractor.predict([tempr,tempg,tempb])
# print(np.max(activation))
# %% 38
# from scipy.io import loadmat
# hadaweightsmodified = loadmat("hadaweightsmodified.mat")
# hadaweightsmodified = hadaweightsmodified['hadaweightsmodified']
# hadaweightsmodified = np.expand_dims(hadaweightsmodified,3)
# weights = generator.layers[3].get_weights()
# weights[0] = hadaweightsmodified
# generator.layers[3].set_weights(weights)
# %%
# # layer = generator.layers[5]
# # feature_extractor = keras.Model(inputs = generator.inputs, outputs = layer.output)
# activation = feature_extractor.predict([tempr,tempg,tempb])
# # plt.imshow(activation[0,:,:,:])
# plt.imshow(activation[0,:,:,:]/np.max(activation[0,:,:,:]))

# %%
def multihada_reconM(ds1,ds2,csizes):
    """
    multistage-hadamard layer reconM
    """
    # keras.__version__
    # ds1: input data size rows
    # ds2: input data size colunms
    # csizes.top: cropping size on top
    # csizes.bottom: cropping size on bottom
    # csizes.left: cropping size on left
    # csizes.right: cropping size on right
    hadalayer1 = Hadamard()
    hadalayer2 = Hadamard()
    hadalayer3 = Hadamard()
    
    conv1layer = layers.Conv2D(1,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.RandomUniform(minval = 0., maxval = 1.),activity_regularizer = regularizers.l2(1e-5))
    rawr = layers.Input(shape = (ds1,ds2,1))
    rawg = layers.Input(shape = (ds1,ds2,1))
    rawb = layers.Input(shape = (ds1,ds2,1))
    hadar1 = hadalayer1(rawr)
    hadar2 = layers.Dropout(0.2)(rawr)
    # hadar = conv1layer(hadar)
    hadar2 = hadalayer2(hadar2)
    hadar3 = layers.Dropout(0.4)(rawr)
    hadar3 = hadalayer3(hadar3)
    hadar = layers.concatenate([hadar1,hadar2,hadar3])
    hadar = conv1layer(hadar)
    
    hadag1 = hadalayer1(rawg)
    hadag2 = layers.Dropout(0.2)(rawg)
    # hadag = conv1layer(hadag)
    hadag2 = hadalayer2(hadag2)
    hadag3 = layers.Dropout(0.4)(rawg)
    hadag3 = hadalayer3(hadag3)
    hadag = layers.concatenate([hadag1,hadag2,hadag3])
    hadag = conv1layer(hadag)
    
    hadab1 = hadalayer1(rawb)
    hadab2 = layers.Dropout(0.2)(rawb)
    # hadab = conv1layer(hadab)
    hadab2 = hadalayer2(hadab2)
    hadab3 = layers.Dropout(0.4)(rawb)
    hadab3 = hadalayer3(hadab3)
    hadab = layers.concatenate([hadab1,hadab2,hadab3])
    hadab = conv1layer(hadab)
    
    hada = layers.concatenate([hadar,hadag,hadab])
    hadacrop = layers.Cropping2D(cropping = ((csizes.top, csizes.bottom), (csizes.left, csizes.right)))(hada)
    hadacropbn = layers.BatchNormalization()(hadacrop)
    reconM = keras.models.Model(inputs = [rawr,rawg,rawb], outputs = hadacropbn)
    # reconM.summary()
    return reconM
# %% 6 
def enhanceModule(ds1,ds2,tsize):
    """
    enhanceM for offset calibration
    """
    # keras.__version__
    # ds1: input data size rows
    # ds2: input data size colunms
    # tsize: target output size
    
    x1 = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x2 = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x3 = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x4 = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x5 = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x6 = layers.Conv2D(256,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x7 = layers.Conv2D(512,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x8 = layers.Conv2D(256,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x9 = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x10 = layers.Conv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x11 = layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x12 = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x13 = layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    x14 = layers.Conv2D(1,(3,3),padding = 'same',activation = 'relu',kernel_initializer = keras.initializers.lecun_normal(seed = None))
    
    rawr = layers.Input(shape = (ds1,ds2,1))
    rawg = layers.Input(shape = (ds1,ds2,1))
    rawb = layers.Input(shape = (ds1,ds2,1))
    
    reconrc = layers.Resizing(tsize, tsize, interpolation = "bicubic")(rawr)
    recongc = layers.Resizing(tsize, tsize, interpolation = "bicubic")(rawg)
    reconbc = layers.Resizing(tsize, tsize, interpolation = "bicubic")(rawb)
    c1r = x1(reconrc)
    c2r = x2(c1r)
    c3r = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_2_1r')(c2r)
    c4r = x3(c3r)
    c5r = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_3_1r')(c4r)
    c6r = x4(c5r)
    c7r = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_4_1r')(c6r)
    c8r = x5(c7r)
    c9r = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_5_1r')(c8r)
    c10r = x6(c9r)
    c11r = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_6_1r')(c10r)
    c12r = layers.Dropout(0.5)(c11r)
    c13r = x7(c12r)
    c14r = layers.Dropout(0.5)(c13r)
    c15r = x8(c14r)
    c16r = x9(c15r)
    c17r = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c16r)
    c18r = layers.concatenate([c17r,c10r], axis = 3)
    c19r = x10(c18r)
    c20r = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c19r)
    c21r = layers.concatenate([c20r,c8r], axis = 3)
    c22r = x11(c21r)
    c23r = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c22r)
    c24r = layers.concatenate([c23r,c6r], axis = 3)
    c25r = x12(c24r)
    c26r = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c25r)
    c27r = layers.concatenate([c26r,c4r], axis = 3)
    c28r = x13(c27r)
    c29r = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c28r)
    c30r = layers.concatenate([c29r,c2r], axis = 3)
    c31r = layers.BatchNormalization()(c30r)
    c32r = x14(c31r)
    
    c1g = x1(recongc)
    c2g = x2(c1g)
    c3g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_2_1g')(c2g)
    c4g = x3(c3g)
    c5g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_3_1g')(c4g)
    c6g = x4(c5g)
    c7g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_4_1g')(c6g)
    c8g = x5(c7g)
    c9g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_5_1g')(c8g)
    c10g = x6(c9g)
    c11g = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_6_1g')(c10g)
    c12g = layers.Dropout(0.5)(c11g)
    c13g = x7(c12g)
    c14g = layers.Dropout(0.5)(c13g)
    c15g = x8(c14g)
    c16g = x9(c15g)
    c17g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c16g)
    c18g = layers.concatenate([c17g,c10g], axis = 3)
    c19g = x10(c18g)
    c20g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c19g)
    c21g = layers.concatenate([c20g,c8g], axis = 3)
    c22g = x11(c21g)
    c23g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c22g)
    c24g = layers.concatenate([c23g,c6g], axis = 3)
    c25g = x12(c24g)
    c26g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c25g)
    c27g = layers.concatenate([c26g,c4g], axis = 3)
    c28g = x13(c27g)
    c29g = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c28g)
    c30g = layers.concatenate([c29g,c2g], axis = 3)
    c31g = layers.BatchNormalization()(c30g)
    c32g = x14(c31g)
    
    c1b = x1(reconbc)
    c2b = x2(c1b)
    c3b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_2_1b')(c2b)
    c4b = x3(c3b)
    c5b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_3_1b')(c4b)
    c6b = x4(c5b)
    c7b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_4_1b')(c6b)
    c8b = x5(c7b)
    c9b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_5_1b')(c8b)
    c10b = x6(c9b)
    c11b = layers.MaxPooling2D(pool_size = (2, 2),padding = 'same',name = 'mp_6_1b')(c10b)
    c12b = layers.Dropout(0.5)(c11b)
    c13b = x7(c12b)
    c14b = layers.Dropout(0.5)(c13b)
    c15b = x8(c14b)
    c16b = x9(c15b)
    c17b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c16b)
    c18b = layers.concatenate([c17b,c10b], axis = 3)
    c19b = x10(c18b)
    c20b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c19b)
    c21b = layers.concatenate([c20b,c8b], axis = 3)
    c22b = x11(c21b)
    c23b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c22b)
    c24b = layers.concatenate([c23b,c6b], axis = 3)
    c25b = x12(c24b)
    c26b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c25b)
    c27b = layers.concatenate([c26b,c4b], axis = 3)
    c28b = x13(c27b)
    c29b = keras.layers.UpSampling2D(size = (2, 2),interpolation = 'bilinear')(c28b)
    c30b = layers.concatenate([c29b,c2b], axis = 3)
    c31b = layers.BatchNormalization()(c30b)
    c32b = x14(c31b)
    
    c32 = layers.concatenate([c32r,c32g,c32b])
    # from keras.models import Model
    # fullgenerator = keras.models.Model(inputs = [rawr,rawg,rawb], outputs = [c32,hadarecon,reconr,recong,reconb,hadar,hadag,hadab])
    enhanceM = keras.models.Model(inputs = [rawr,rawg,rawb], outputs = c32)
    # enhanceM.summary()
    return enhanceM


