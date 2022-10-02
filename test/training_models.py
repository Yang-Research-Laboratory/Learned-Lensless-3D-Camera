# %% define model input size
ds1=1376 # input image size
ds2=2064 # input image size
csize=383 # training target size
rsize=256 # enhancement module output size
reconM=reconMRGB(ds1,ds2,csize) # generate reconstruction module from template
reconM.summary()
# %% load training datasets
from scipy.io import loadmat
datav1=loadmat("datasets\\data.mat")
Xr=datav1['Xr']
Xr=Xr.astype('float32')
Xr=np.expand_dims(Xr,3)
Xg=datav1['Xg']
Xg=Xg.astype('float32')
Xg=np.expand_dims(Xg,3)
Xb=datav1['Xb']
Xb=Xb.astype('float32')
Xb=np.expand_dims(Xb,3)
# Y=loadmat("Y.mat")
Y=datav1['Y']
Y=Y.astype('float32')
# %% compile reconM module
# generator.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
reconM.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['mae',ssim,psnr])
# generator.compile(optimizer='rmsprop',loss=custom_objective_lpips,metrics=['mae'])
# generator.compile(optimizer='rmsprop',loss=custom_objective,metrics=['mae'])
# generator.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['mae'])
# %% training reconM
history=reconM.fit([Xr,Xg,Xb],Y,epochs=120,batch_size=1,verbose=1,shuffle=True)
# %% generate full generator from template
fullgenerator=fullgeneratorRGB(ds1,ds2,csize,rsize)
fullgenerator.summary()
# %% compile full generator with customized loss function and weights
# fullgenerator.compile(optimizer='rmsprop',loss=custom_objective,metrics=['mae'],run_eagerly=True)
fullgenerator.compile(optimizer='rmsprop',loss=custom_objective,metrics=['mae'],run_eagerly=True)
# fullgenerator.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['mae'],run_eagerly=True)
# generator.compile(optimizer='rmsprop',loss=custom_objective2,metrics=['mae'])
# %% load enhancement module training dataset
from scipy.io import loadmat
datav1=loadmat("datasets\\data_Yenhance.mat")
Yenhance=datav1['Yenhance']
Yenhance=Yenhance.astype('float32')
# %% 19 set layer trainable
for lid in range(0,6): # assign Hadamard layers weights
    weights=reconM.layers[lid].get_weights()
    fullgenerator.layers[lid].set_weights(weights)
fullgenerator.layers[3].trainable = False
fullgenerator.layers[4].trainable = False
fullgenerator.layers[5].trainable = False
# fullgenerator.layers[8].trainable = False
# %% U-net enhancement network training
history=fullgenerator.fit([Xr,Xg,Xb],Yenhance,epochs=5,batch_size=1,verbose=1,shuffle=True)
# %% configure discriminator
channels=3
dsc=256
drsize=256
discriminator=discriminator_v2(channels,dsc,drsize)
discriminator.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics='mse')
discriminator.summary()
# %% pre-train discriminator from
bsize=20
disX=np.zeros((bsize*2,drsize,drsize,3))
for pid in range(0,bsize):
    generated_images=fullgenerator.predict([Xr[pid:pid+1],Xg[pid:pid+1],Xb[pid:pid+1]])
    disX[pid]=generated_images[0]
    print(pid)
for pid in range(bsize,bsize*2):
  disX[pid]=Yenhance[pid-bsize]
disY=np.zeros((bsize*2,1))
for pid in range(0,bsize):
  disY[pid]=1
# %% pretrain discriminator
history=discriminator.fit(disX,disY,epochs=5,batch_size=5,verbose=1,shuffle=True)
# %% train full generator with adversarial loss
iterations=150
import time
batch_size=1
dsize=256
start=0
# th=0.05
for step in range(iterations):
    stop=start+batch_size
    generated_images=np.zeros((batch_size,dsize,dsize,3))
    for giter in range(0,batch_size):
        generated_images[giter]=fullgenerator.predict([Xr[start+giter:start+giter+1],Xg[start+giter:start+giter+1],Xb[start+giter:start+giter+1]])    
    # generated_images=fullgenerator.predict([Xr[start:stop],Xg[start:stop],Xb[start:stop]])
    real_images=Yenhance[start:stop]
    combined_images=np.concatenate([generated_images,real_images])
    labels=np.concatenate([np.ones((batch_size,1)),np.zeros((batch_size,1))])
    labels+=0.05*np.random.random(labels.shape)
    d_loss=discriminator.train_on_batch(combined_images,labels)
    g_loss=fullgenerator.train_on_batch([Xr[start:stop],Xg[start:stop],Xb[start:stop]],Yenhance[start:stop])
    start+=batch_size
    if start>len(Xr)-batch_size:
        start=0
    if step % 10 ==0:
        print('iteration:',step,'discriminator loss:',d_loss,'fullgenerator loss:', g_loss)
    # if (g_loss[1]<th and step>10):
    #     print('break on iteration',step)
    #     break
# %% set Hadamard layer as trainable
fullgenerator.layers[3].trainable = True
fullgenerator.layers[4].trainable = True
fullgenerator.layers[5].trainable = True
# generator.layers[8].trainable = True
# %% train full generator with adversarial loss
iterations=150
import time
batch_size=1
dsize=256
start=0
# th=0.05
for step in range(iterations):
    stop=start+batch_size
    generated_images=np.zeros((batch_size,dsize,dsize,3))
    for giter in range(0,batch_size):
        generated_images[giter]=fullgenerator.predict([Xr[start+giter:start+giter+1],Xg[start+giter:start+giter+1],Xb[start+giter:start+giter+1]])    
    # generated_images=fullgenerator.predict([Xr[start:stop],Xg[start:stop],Xb[start:stop]])
    real_images=Yenhance[start:stop]
    combined_images=np.concatenate([generated_images,real_images])
    labels=np.concatenate([np.ones((batch_size,1)),np.zeros((batch_size,1))])
    labels+=0.05*np.random.random(labels.shape)
    d_loss=discriminator.train_on_batch(combined_images,labels)
    g_loss=fullgenerator.train_on_batch([Xr[start:stop],Xg[start:stop],Xb[start:stop]],Yenhance[start:stop])
    start+=batch_size
    if start>len(Xr)-batch_size:
        start=0
    if step % 10 ==0:
        print('iteration:',step,'discriminator loss:',d_loss,'fullgenerator loss:', g_loss)
    # if (g_loss[1]<th and step>10):
    #     print('break on iteration',step)
    #     break

# %%
# psf=loadmat("psfweights.mat")
# psf=psf['psfweights']
# # psf=np.expand_dims(psf,2)
# psf=np.expand_dims(psf,3)
# weights=generator.layers[3].get_weights()
# weights[0]=psf
# generator.layers[3].set_weights(weights)
# generator.layers[4].set_weights(weights)
# generator.layers[5].set_weights(weights)
