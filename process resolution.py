# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:09:25 2016

@author: dion
"""

import numpy as np
import scipy.ndimage as img
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches
from skimage import restoration
from mpl_toolkits.axes_grid1 import make_axes_locatable

import TwoDTomography as tm
import data_prospa
import ConfigParser

datafolder = "data/7 - ring 1mm/"

#training sample description
convolver = 10*np.ones((30,30))
convolver[10:20,10:20]=0



data2D = data_prospa.read_2d_file(datafolder+"dataIMG.2d")
data3D=data_prospa.read_3d_file(datafolder+"data.3d")
#params=ConfigParser.ConfigParser()
#params.read(datafolder+"acqu.par")

#implement gating on 3d dataset
data2D*=0.01
data3D*=0.01
processedout=np.sum(data3D.real[:,:,0:],2)
processedout*= (data2D.real.max() / processedout.max())
deconv80=restoration.wiener(processedout,convolver,200,clip=False)


data2D=data_prospa.read_2d_file("data/6 - ring 2mm/dataIMG.2d")
data2D=0.01*data2D.real
deconv40=restoration.wiener(data2D,convolver[::2,::2],200,clip=False)


fig1,ax=plt.subplots(nrows=1,ncols=5,squeeze=False,figsize=(15,5))
ax[0,0].matshow(convolver)
ax[0,0].set_title('original')
ms11=ax[0,1].matshow(processedout)
ax[0,1].set_title('imagedata80')
fixclim=ms11.get_clim()

ms12=ax[0,2].matshow(data2D)
ms12.set_clim(fixclim)
ax[0,2].set_title('imagedata40')

ms13=ax[0,3].matshow(deconv80)
ax[0,3].set_title('Deconv80')
#ms13.set_clim(fixclim)

ms14=ax[0,4].matshow(deconv40)
ax[0,4].set_title('Deconv40')

restorebalance=0.05
reconst80=restoration.wiener(processedout,deconv80,restorebalance/2,clip=False)
reconst40=restoration.wiener(data2D,deconv40,restorebalance/2,clip=False)

#fig2,ax=plt.subplots(nrows=1,ncols=2,squeeze=False,figsize=(15,5))
#ms11=ax[0,0].matshow(reconst80)
#ax[0,0].set_title('reconst80')
#fixclim=ms11.get_clim()
#
#ms12=ax[0,1].matshow(reconst40)
#ms12.set_clim(fixclim)
#ax[0,1].set_title('reconst40')

#Save psf
np.save('psf80.npy', deconv80[22:62,22:62])
np.save('psf40.npy', deconv40[11:31,11:31])
#
fig3,ax=plt.subplots(nrows=1,ncols=2,squeeze=False,figsize=(15,5))
ms11=ax[0,0].matshow(deconv80[22:62,22:62])
ax[0,0].set_title('psf80')
fixclim=ms11.get_clim()
#
ms12=ax[0,1].matshow(deconv40[11:31,11:31])
ms12.set_clim(fixclim)
ax[0,1].set_title('psf40')

