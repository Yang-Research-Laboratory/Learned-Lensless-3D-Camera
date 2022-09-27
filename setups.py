# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:35:36 2022

@author: fengt
"""

# %% 1
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import keras
# from keras import layers
from keras import backend as K
import keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
latent_dim=16
height=64
width=64
channels=3
from tensorflow.keras import regularizers

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

