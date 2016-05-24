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

datafolder = "data/8 - W 2mm/"

#training sample description
#convolver = 10*np.ones((30,30))
#convolver[10:20,10:20]=0

convolver=np.load('psf40.npy')
convolver=5.0*img.imread(datafolder+"pattern2mm.png",flatten=True)/255.0
convolver+=5

data2D = data_prospa.read_2d_file(datafolder+"dataIMG.2d")
data3D=data_prospa.read_3d_file(datafolder+"data.3d")
#params=ConfigParser.ConfigParser()
#params.read(datafolder+"acqu.par")

#implement gating on 3d dataset
data2D*=0.01
data3D*=0.01
processedout=np.sum(data3D.real[:,:,0:],2)
processedout*= (data2D.real.max() / processedout.max())
deconv=restoration.wiener(processedout,convolver,3e-1,clip=False)
deconv=restoration.wiener(processedout,convolver,200,clip=False)

#np.save('psf40.npy', deconv)

fig1,ax=plt.subplots(nrows=1,ncols=4,squeeze=False,figsize=(15,5))
ms11=ax[0,0].matshow(data2D.real)
ax[0,0].set_title('Sumimage')
fixclim=ms11.get_clim()

ms12=ax[0,1].matshow(processedout)
ms12.set_clim(fixclim)
ax[0,1].set_title('Gateimage')

ms13=ax[0,2].matshow(convolver)
ax[0,2].set_title('Convolver')
#ms13.set_clim(fixclim)

ms14=ax[0,3].matshow(deconv)
ax[0,3].set_title('Wiener Deconv')

fig1,ax=plt.subplots(nrows=1,ncols=3,squeeze=False,figsize=(15,5))
deconv2 = deconv.copy()
#deconv2=deconv[12:32,12:32]
deconv2[deconv2<0.1*np.amax(deconv2)] = 0 
ms11=ax[0,0].matshow(deconv2)
ax[0,0].set_title('W sweetspot')
fixclim=ms11.get_clim()

#ms12=ax[0,1].matshow(np.load('psf40.npy'))
ms12=ax[0,1].matshow(restoration.wiener(processedout,deconv2,0.05,clip=False))

ringdata=data_prospa.read_2d_file('data/6 - ring 2mm/dataIMG.2d')
ringdata=0.01*ringdata.real

ax[0,2].matshow(restoration.wiener(ringdata,deconv2,balance=0.05,clip=False))
ax[0,2].set_title('ring reconstruction')
