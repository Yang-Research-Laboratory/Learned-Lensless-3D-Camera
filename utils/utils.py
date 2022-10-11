# %%

import tensorflow as tf
import keras
from keras import backend as K
import keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
# %%
"""
define tensor Fourier transform
"""
def ftf(x):
    x = tf.cast(x,tf.complex64)
    x = tf.signal.fft2d(x)
    x = tf.abs(x)
    x = tf.cast(x,tf.float32)
    return x
def iftf(x):
    x = tf.cast(x,tf.complex64)
    x = tf.signal.ifft2d(x)
    x = tf.abs(x)
    x = tf.cast(x,tf.float32)
    return x
# %%
"""
metric functions PNSR and SSIM
"""
def ssim(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
def psnr(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis = -1)))) / 2.303
# %%
"""
custom_objectives for photorealistic generator model
"""
# def custom_objective(y_pred,y_true,weight1,weight2,weight3,weight4):
def custom_objective(y_pred,y_true):
    # y_pred: generated reconstruction
    # y_true: target objects
    # weight1: weight of MSE loss
    # weight2: weight of perceptual loss
    # weight3: weight of total variation loss
    # weight4: weight of adversarial loss
    # weights parameters can be adjusted based on experience
    y_predvgg = tf.image.resize(y_pred, (224,224), method = tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio = False,antialias = False, name = None)
    y_truevgg = tf.image.resize(y_true, (224,224), method = tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio = False,antialias = False, name = None)
    tv = tf.image.total_variation(y_pred)
    y1 = vgg_model(y_predvgg)
    y2 = vgg_model(y_truevgg)
    y3 = discriminator(y_pred)
    loss2 = K.mean(K.square(y2 - y1), axis = [1,2,3])
    loss1 = K.mean(K.square(y_pred - y_true), axis = [1,2,3])
    weight1 = 1.5
    weight2 = 0.1
    weight3 = 5e-7
    weight4 = 8e-3
    loss = loss1*weight1 + loss2*weight2 + tv*weight3 + y3*weight4
    # print('loss1 = ',str(K.get_value(loss1*weight1)),'loss2 = ',str(K.get_value(loss2*weight2)),'tv = ',str(K.get_value(tv*weight3)),'y3 = ',str(K.get_value(y3*weight4)))
    return loss
# %%
"""
Lambda layer functions
"""
def rnorm(ip):
    # x[0] = x[0]/K.max(x[1])
    x = ip[0]
    y = ip[1]
    # print(K.max(y))
    res = x/K.max(y)*1.1
    return res
def sepa(x,y):
    # x[0] = x[0]/K.max(x[1])
    # x = ip[0]
    # y = ip[1]
    return x[0,:,:,y]
# %%
"""
pretrained VGG-16 net
"""
from keras.applications.vgg16 import VGG16
vgg_model = VGG16(weights = 'imagenet',include_top = False)
vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False
# %%
"""
if using LPIPS metrics
"""
# import sys
# sys.path
# print(sys.path)
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()
# tf.enable_eager_execution()
# %% 24
# weights = raw.get_weights()
def plotweights(model,lid):
    """
    plot layer weights of trained models
    """
    # model: trained models name
    # lid: layer number
    weights = model.layers[lid].get_weights()
    temp = weights[0]
    plt.imshow(temp[:,:,0,0])
    # plt.imshow(temp[0,:,:,0])
    # plt.clim(-1,1)
    plt.colorbar()
# %% 26
def plotmodel(model):
    """
    plot model layers
    """
    tf.keras.utils.plot_model(
        model,
        to_file = "model.png",
        show_shapes = True,
        show_dtype = True,
        show_layer_names = True,
        rankdir = "TB",
        expand_nested = True,
        dpi = 96,
        layer_range = None,
    )
# %%
def reconout(reconM,of1,of2,tcenter1,tcenter2,lid,rawr,rawg,rawb):
    """
    reconM for 3D imaging
    """
    # reconM: trained reconstruction models at multiple distances
    # of1,of2: offset for each distance along x and y
    # tcenter1,tcenter2: center of FOV coordinates along x and y
    # lid: layer id for feature extraction, for RGB image, lid = 6, for single color image lid = 4
    center1 = tcenter1-np.int(of1)
    center2 = tcenter2-np.int(of2)
    layer = reconM.layers[lid]
    feature_extractor = keras.Model(inputs = reconM.inputs, outputs = layer.output)
    activation = feature_extractor.predict([rawr,rawg,rawb])
    out = activation[0,:,:,:]/np.max(activation[0,:,:,:])
    out = out[center1-125:center1+175,center2-150:center2+150,:]
    out = out/np.max(out)
    # out = out*3
    return out
# %%
def generatorout(enhanceM,inr,ing,inb):
    """
    enhanceM for photorealistic post processing
    """
    # enhanceM: trained enhancement module
    # inr: input image red color channel
    # ing: input image green color channel
    # inb: input image blue color channel
    out = enhanceM.predict([inr,ing,inb])
    out = out/np.max(out)
    return out


