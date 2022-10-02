import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
# %% load datasets
from scipy.io import loadmat
datav1=loadmat("datasets//vdata_seethrough.mat")
# import h5py
# datav1 = h5py.File('datav7.mat')
# import mat73
# datav1 = mat73.loadmat('datav7_2.mat')
Xvr=datav1['Xvr']
Xvr=Xvr.astype('float32')
# Xvr=np.expand_dims(Xvr,0)
Xvr=np.expand_dims(Xvr,3)
Xvg=datav1['Xvg']
Xvg=Xvg.astype('float32')
# Xvg=np.expand_dims(Xvg,0)
Xvg=np.expand_dims(Xvg,3)
Xvb=datav1['Xvb']
Xvb=Xvb.astype('float32')
# Xvb=np.expand_dims(Xvb,0)
Xvb=np.expand_dims(Xvb,3)
pid=0
rawr=Xvr[pid]
rawr=np.expand_dims(rawr,0)
rawg=Xvg[pid]
rawg=np.expand_dims(rawg,0)
rawb=Xvb[pid]
rawb=np.expand_dims(rawb,0)
# %% load pre-trained reconstruction modules
reconM1=keras.models.load_model('reconM_1.h5', custom_objects={'Hadamard': Hadamard})
reconM2=keras.models.load_model('reconM_2.h5', custom_objects={'Hadamard': Hadamard})
reconM3=keras.models.load_model('reconM_3.h5', custom_objects={'Hadamard': Hadamard})
reconM4=keras.models.load_model('reconM_4.h5', custom_objects={'Hadamard': Hadamard})
reconM5=keras.models.load_model('reconM_5.h5', custom_objects={'Hadamard': Hadamard})
reconM6=keras.models.load_model('reconM_6.h5', custom_objects={'Hadamard': Hadamard})
reconM7=keras.models.load_model('reconM_7.h5', custom_objects={'Hadamard': Hadamard})
reconM8=keras.models.load_model('reconM_8.h5', custom_objects={'Hadamard': Hadamard})
reconM9=keras.models.load_model('reconM_9.h5', custom_objects={'Hadamard': Hadamard})
reconM10=keras.models.load_model('reconM_10.h5', custom_objects={'Hadamard': Hadamard})
reconM11=keras.models.load_model('reconM_11.h5', custom_objects={'Hadamard': Hadamard})
reconM12=keras.models.load_model('reconM_12.h5', custom_objects={'Hadamard': Hadamard})
reconM13=keras.models.load_model('reconM_13.h5', custom_objects={'Hadamard': Hadamard})
reconM14=keras.models.load_model('reconM_14.h5', custom_objects={'Hadamard': Hadamard})
# reconM15=keras.models.load_model('15\\reconM_15.h5', custom_objects={'Hadamard': Hadamard})
# reconM=reconM10
# %% load pre-trained enhancement modules
fullgenerator = keras.models.load_model('fullgenerator.h5', custom_objects = {'Hadamard': Hadamard,'custom_objective': custom_objective})
# %% define the test field of view
ds1 = 300 # test FOV x dimension #pixels
ds2 = 300 # test FOV dimension #pixels
tsize = 256 # enhancement module output size
enhanceM = enhanceModule(ds1,ds2,tsize) # generate enhancement model from definition
enhanceM.summary()
# %% assign pre-trained weights of enhancement module layers
for lid in range(12,len(fullgenerator.layers)):
    weights=fullgenerator.layers[lid].get_weights()
    enhanceM.layers[lid-6].set_weights(weights)
# %% run reconM and enhanceM for pre-trained distances
reconMlist = [reconM1,reconM2,reconM3,reconM4,reconM5,reconM6,reconM7,reconM8,reconM9,reconM10,reconM11,reconM12,reconM13,reconM14]
recon_stack = np.zeros((15,ds1,ds2,3)) # reconM results
generated_stack = np.zeros((15,tsize,tsize,3)) # enhanceM post processing results
# generated_stack3 = np.zeros((15,256,256,3))
ofs = loadmat("datasets//ofs.mat")
ofx = ofs['ofx']
ofy = ofs['ofy']
# ofx = np.ndarray.tolist(ofx)
# ofy = np.ndarray.tolist(ofy)
tcenter1 = 730
tcenter2 = 1010
lid=6
for rid in range(0,14):
    temp = reconout(reconMlist[rid],ofx[rid],ofy[rid],tcenter1,tcenter2,lid,rawr,rawg,rawb)
    recon_stack[rid] = temp
    inr = temp[:,:,0]
    inr = np.expand_dims(inr,0)
    inr = np.expand_dims(inr,3)
    ing = temp[:,:,1]
    ing = np.expand_dims(ing,0)
    ing = np.expand_dims(ing,3)
    inb = temp[:,:,2]
    inb = np.expand_dims(inb,0)
    inb = np.expand_dims(inb,3)
    generated_stack[rid] = generatorout(enhanceM,inr,ing,inb)
# %% plot saved reconstructions in FOV
for rid in range(0,14):
    plt.figure()
    plt.imshow(recon_stack[rid])
    plt.figure()
    plt.imshow(generated_stack[rid])
# %%
# from scipy.io import savemat
# savemat("recon_stack.mat", {"recon_stack": recon_stack})
# savemat("generated_stack.mat", {"generated_stack": generated_stack})
