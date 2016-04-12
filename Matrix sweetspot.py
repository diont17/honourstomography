# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:58:05 2016

@author: Dion
"""

import numpy as np
import scipy as scipy
import matplotlib.pylab as plt
import numpy.linalg as linalg
from scipy.ndimage.filters import gaussian_filter
from scipy.signal.windows import gaussian as gaussian_window

import slideTomography_1d as tm
#reload(tm)
#%%

sweetSptSize=70
sweetSptPadding=15

sweetSpot= np.zeros(sweetSptSize + 2*sweetSptPadding)
sweetSpot[sweetSptPadding:-sweetSptPadding]=1
sweetSpot=gaussian_filter(sweetSpot,5)
#sweetSpot=gaussian_window(100,40)

t1d1= tm.Tomography_1D(sweetSpot)
t1d2=tm.Tomography_1D(sweetSpot)
t1d3=tm.Tomography_1D(sweetSpot)

#%%Training data 1: Block with lots of holes, different sizes

calSize=500
calSmp1=2*np.ones(calSize)
calSmp1[40:50]=0
calSmp1[60:70]=0
calSmp1[80:100]=0
calSmp1[120:150]=0
calSmp1[180:220]=0
calSmp1[260:300]=0
calSmp1[450:500]=0

sg=3
calSmp1=gaussian_filter(calSmp1,sg)

calSig1=t1d1.calcSignal(calSmp1)

#Training data 2: Solid block
calSmp2=np.zeros(calSize)
calSmp2[50:450]=2
calSmp2=gaussian_filter(calSmp2,sg)

calSig2=t1d2.calcSignal(calSmp2)

#Training data 3: Block with one hole
#
calSmp3=np.zeros(calSize)
calSmp3[100:200]=2
calSmp3[250:400]=1


calSig3=t1d2.calcSignal(calSmp3)
calSize=len(calSig3)
#%%Train t1d
np.random.seed(2)
noisesize=0.05*max(calSig2)

#storednoise=noisesize-2*noisesize*np.random.rand(calSize)
#t1d1.addTrainingData(np.append(np.zeros(50),calSmp1),calSig1+storednoise)
#storednoise=noisesize-2*noisesize*np.random.rand(calSize)
#t1d1.addTrainingData(np.append(np.zeros(50),calSmp2),calSig2+storednoise)
storednoise=noisesize-2*noisesize*np.random.rand(calSize)
#t1d1.addTrainingData(np.append(np.zeros(50),calSmp3),calSig3+storednoise)

t1d2.addTrainingData(np.append(np.zeros(50),calSmp2),calSig2+storednoise)

#Setup an array of t1ds, which will be trained with data:signal offsets
shiftrange=50
shiftedt1ds=list()


#for i in xrange(shiftrange):
#    shiftedt1ds.append(tm.Tomography_1D(sweetSpot))
##    shiftcalSmp1=np.append(np.zeros(50+shiftrange/2),calSmp1)[i:shiftrange+len(calSmp1)]
#    shiftcalSmp2=np.append(np.zeros(50+shiftrange/2),calSmp2)[i:shiftrange+len(calSmp2)]
##    shiftcalSmp3=np.append(np.zeros(50+shiftrange/2),calSmp3)[i:shiftrange+len(calSmp3)]
#
#    storednoise=noisesize-noisesize*2*np.random.rand(calSize)
#    shiftedt1ds[i].addTrainingData(shiftcalSmp2,calSig2+storednoise)
#    print 'optimized {0}'.format(i)
##    storednoise=noisesize-noisesize*2*np.random.rand(calSize)
##    shiftedt1ds[i].addTrainingData(shiftcalSmp1,calSig1+storednoise)
##    storednoise=noisesize-noisesize*2*np.random.rand(calSize)
##    shiftedt1ds[i].addTrainingData(shiftcalSmp3,calSig3+storednoise)
    
CAin=np.load('outCA.npz')
for i in xrange(shiftrange):
    key='arr_{0}'.format(i)
    shiftedt1ds.append(tm.Tomography_1D(sweetSpot))
    shiftedt1ds[i].CA=CAin[key]

#%%Reconstruction Tests

smpSize=500
smpPadding=2

# Sine function
sinesmp  = np.zeros([smpSize])
sinesmp[smpPadding:-smpPadding] =  2*np.sin(np.linspace(0,7*np.pi,smpSize-2*smpPadding))+2

rounsample=np.array([0,0,0,0,0.5,0.6,0.8,1,1.5,2,3,4,5,7,9,12,15,18,19,20,21,21.5,22,22.5,23,23.2,23.1,23.1,23.2,23.3,23.2,23.1,35,23.2,23.2,
                     22.5,22,21.5,21,20.5,20,19,17,15,12,9,7,5,4,3,2,1.5,1,1,0.8,0.4,0,0,0,0])

rounsmp=np.zeros(60*3)
fill=0
rounsmp=np.zeros(smpSize)
for i in xrange(60):
    rounsmp[fill:fill+4]=rounsample[i]
    rounsmp[fill+200:fill+204]=rounsample[i]
    fill+=4
    
smp=np.linspace(0,2,300)
smp=gaussian_filter(calSmp3,2)

smp=calSmp1
testsig = t1d2.calcSignal(smp)

noisesize=0.1*max(testsig)

storednoise=noisesize-noisesize*2*np.random.rand(600)
testsig+=storednoise[:len(testsig)]
#rec = t1d1.reconstruct(testsig)
rec2= t1d2.reconstruct(testsig)

avgrec= np.zeros_like(rec2)
shiftweight=np.exp(-0.5*((np.arange(0,100)-(50))/(20))**2)

shiftrecs=np.zeros((shiftrange,len(rec2)))
rawshiftrecs=np.zeros((shiftrange,len(rec2)))

for i in xrange(shiftrange):
    rawshiftrecs[i]=shiftedt1ds[i].reconstruct(testsig)
    padrec=np.zeros(shiftrange+len(rec2))
    padrec[shiftrange/2:len(rec2)+shiftrange/2]=rawshiftrecs[i]

    shiftrecs[i]=padrec[(shiftrange-i):(shiftrange-i)+len(rec2)]
    if i>0: 
        avgrec+=shiftrecs[i]
        
avgrec*=1.0/(shiftrange)


#%% Draw Graphs
plt.figure()
plt.plot(np.arange(50,50+len(smp)),smp,label='o')
plt.plot(testsig*0.01,label='s/100')
#plt.plot(rec,label='rec (multiple trainings)')
plt.plot(rec2,label='rec2 (1 training)')
plt.plot(avgrec,label='rec (average of shifted trainings)')
plt.legend()

plt.figure()
plt.plot(t1d2.CA,label='opt')
plt.plot(t1d2.CA0,label='lstsq')
plt.legend()

plt.figure()
t1d2.CA=t1d2.CA0
rec2=t1d2.reconstruct(testsig)
plt.plot(np.arange(50,50+len(smp)),smp,label='o')
plt.plot(testsig*0.01,label='s/100')
#plt.plot(rec,label='rec (multiple trainings)')
plt.plot(rec2,label='rec2 lstsq')

plt.figure()
for m in shiftrecs:
    plt.plot(m)
plt.plot(np.arange(50,50+len(smp)),smp,label='o')
plt.plot(shiftrecs[0],'.-',label=0)
plt.plot(shiftrecs[shiftrange/2],'-',label=(shiftrange/2))
plt.plot(shiftrecs[shiftrange-1],'.-',label=i)
plt.plot(avgrec,'^',label='avg')
plt.legend()

#%% Output Optimized CA arrays
np.savez('outCA.npz',*[shiftedt1ds[x].CA for x in xrange(shiftrange)])
    