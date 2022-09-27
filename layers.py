# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:35:00 2022

@author: fengt
"""
import keras
import tensorflow as tf
# %% 4
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
