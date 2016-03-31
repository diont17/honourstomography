# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:58:05 2016

@author: Dion
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:42:33 2016

@author: dion
"""

import numpy as np
import scipy as sci
import matplotlib.pylab as plt
import numpy.linalg as linalg
from scipy.ndimage.filters import gaussian_filter

import slideTomography_1d as tm
#reload(tm)

swSptSize=80
swSptPadding=10

sweetSpot= np.zeros(swSptSize + 2*swSptPadding)
sweetSpot[swSptPadding:-swSptPadding]=1
#sweetSpot=gaussian_filter(sweetSpot,5)

t1d= tm.Tomography_1D(sweetSpot)

calSize=300
calSmp=np.zeros(calSize)
calSmp[40:260]=2
calSmp[150:200]=0
calSmp[70:120]=0
#
#calSmp[100:130]=1

sg=10
calSmp=gaussian_filter(calSmp,sg)

np.random.seed(2)
storednoise=40-80.0*np.random.rand(600)

calSig=t1d.calcSignal(calSmp)
calSig+=storednoise[:len(calSig)]
t1d.calibrate(calSmp,calSig)


smpSize=500
smpPadding=60

# Sine function
storednoise=1-2*np.random.rand(smpSize)
smp  = np.zeros([smpSize])
smp[smpPadding:-smpPadding] =  np.sin(np.linspace(0,20*np.pi,smpSize-2*smpPadding))

sinesig = t1d.calcSignal(smp)
#signal+=storednoise
rec = t1d.reconstruct(sinesig)
plt.plot(smp,label='o')
plt.plot(rec,label='rec')
plt.legend()
