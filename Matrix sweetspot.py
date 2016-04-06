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
from scipy.signal.windows import gaussian as gaussian_window

import slideTomography_1d as tm
#reload(tm)

swSptSize=70
swSptPadding=15

sweetSpot= np.zeros(swSptSize + 2*swSptPadding)
sweetSpot[swSptPadding:-swSptPadding]=1
sweetSpot=gaussian_filter(sweetSpot,5)
#sweetSpot=gaussian_window(100,40)


t1d= tm.Tomography_1D(sweetSpot)
t1d2=tm.Tomography_1D(sweetSpot)
t1d3=tm.Tomography_1D(sweetSpot)

calSize=500
calSmp=2*np.ones(calSize)
calSmp[40:50]=0
calSmp[60:70]=0
calSmp[80:100]=0
calSmp[120:150]=0
calSmp[180:220]=0
calSmp[260:300]=0
calSmp[450:500]=0

sg=3
calSmp=gaussian_filter(calSmp,sg)

np.random.seed(2)
noisesize=0.1
storednoise=noisesize-noisesize*2*np.random.rand(600)

calSig=t1d.calcSignal(calSmp)
calSig+=storednoise[:len(calSig)]
t1d.calibrate(np.append(np.zeros(50),calSmp),calSig)

calSmp2=np.zeros(calSize)
calSmp2[50:450]=2
calSmp2=gaussian_filter(calSmp2,sg)

storednoise=noisesize-noisesize*2*np.random.rand(600)
calSig2=t1d2.calcSignal(calSmp2)
calSig2+=storednoise[:len(calSig2)]
t1d2.calibrate(np.append(np.zeros(50),calSmp2),calSig2)

calSmp3=np.zeros(calSize)
calSmp3[100:200]=2
calSmp3[300:400]=2

#storednoise=noisesize-noisesize*2*np.random.rand(600)
#calSig3=t1d3.calcSignal(calSmp3)
#calSig3+=storednoise[:len(calSig3)]
#t1d3.calibrate(np.append(np.zeros(50),calSmp3),calSig3)

smpSize=500
smpPadding=2

# Sine function
smp  = np.zeros([smpSize])
smp[smpPadding:-smpPadding] =  np.sin(np.linspace(0,10*np.pi,smpSize-2*smpPadding))

rounsample=np.array([0,0,0,0,0.5,0.6,0.8,1,1.5,2,3,4,5,7,9,12,15,18,19,20,21,21.5,22,22.5,23,23.2,23.1,23.1,23.2,23.3,23.2,23.1,28,23.2,23.2,22.5,22,21.5,
                     21,20.5,20,19,17,15,12,9,7,5,4,3,2,1.5,1,1,0.8,0.4,0,0,0,0])

smp=np.zeros(60*3)                     
fill=0
for i in range(60):
    smp[fill:fill+2]=rounsample[i]
    fill+=2
#    
smp=calSmp2

noisesize=5
storednoise=noisesize-noisesize*2*np.random.rand(600)

testsig = t1d.calcSignal(smp)
testsig+=storednoise[:len(testsig)]
rec = t1d.reconstruct(testsig)
rec2=t1d2.reconstruct(testsig)
#rec3=t1d3.reconstruct(testsig)
avgrec=((rec+rec2)/2.0)


plt.plot(np.arange(50,50+len(smp)),smp,label='o')
plt.plot(testsig*0.01,label='s/100')
plt.plot(rec,label='rec')
plt.plot(rec2,label='rec2')
#plt.plot(rec3,label='rec3')
plt.plot(avgrec,label='avg')

plt.legend()


#plt.plot(calSig2/40,label='S/40')
#plt.plot(np.append(np.zeros(50),calSmp),label='O')
#plt.plot(t1d2.reconstruct(calSig),label='R')
#plt.legend()