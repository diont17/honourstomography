# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 14:45:18 2016

@author: dion
"""

import numpy as np
import scipy.ndimage as img
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import cm

import TwoDTomography as tm


def addNoise2D(arr, noisepercent):
    arrx= arr.shape[1]
    arry= arr.shape[0]
    noise = noisepercent - 2*noisepercent*np.max(arr)*np.random.rand(arry,arrx)
    return arr+noise

np.random.seed(2)
size=20
x, y = np.float64(np.mgrid[-size/2 + 1:size/2 + 1, -size/2 + 1:size/2 + 1])
x-=0.5
y-=0.5
#0.9 for narrow sweetspot
#0.4 for wide sweetspot
std=0.2

sweetSpot=np.exp(- ((x**2 + y**2)/ 2*std**2))
sweetSpot*=(1.0/np.max(sweetSpot))
#sweetSpot=np.repeat(np.repeat(sweetSpot,2,axis=0),2,axis=1)
#sweetSpot=np.ones((6,6))
t2d=tm.Tomographer_2d(sweetSpot)

#smp=img.imread('whitesquare.png',flatten=True)[::80,::80]
smp1= np.array([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])
smp1= np.array([[0,0,0,0,1,1,1,1,1,1,1,1],[0,0,0,0,1,1,1,1,1,1,1,1],[0,0,0,0,1,1,1,1,1,1,1,1],[0,0,0,0,1,1,1,1,1,1,1,1],
                [1,1,1,1,0,0,0,0,1,1,1,1],[1,1,1,1,0,0,0,0,1,1,1,1],[1,1,1,1,0,0,0,0,1,1,1,1],[1,1,1,1,0,0,0,0,1,1,1,1],
                [1,1,1,1,1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1,0,0,0,0]])

smp2=10*np.array([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])
smp3=img.imread('whitering.png', flatten=True)[::10,::10]
smp4=img.imread('stripes.png', flatten=True) [::10,::2]
smp3/=255.0
smp4/=255.0
smp1=np.repeat(np.repeat(smp1,2,axis=0),2,axis=1)
#smp2=np.repeat(np.repeat(smp2,2,axis=0),2,axis=1)

sig1=t2d.calcSignal(smp1)
sig2=t2d.calcSignal(smp2)
sig3=t2d.calcSignal(smp3)
sig4=t2d.calcSignal(smp4)

#plt.matshow(padsmp)
#plt.matshow(smp,cmap=cm.gray)
#plt.title('sample')
#plt.figure()
#plt.matshow(sig,cmap=cm.gray)
#plt.title('signal')

#t2d.addTrainingData(smp1,addNoise2D(sig1,0.01))
t2d.addTrainingData(smp1,addNoise2D(sig1,0.03))
#t2d.addTrainingData(smp1,addNoise2D(sig1,0.05))
#
#t2d.addTrainingData(smp2,addNoise2D(sig2,0.01))
#t2d.addTrainingData(smp2,addNoise2D(sig2,0.03))
#t2d.addTrainingData(smp2,addNoise2D(sig2,0.05))
t2d.lam=0
t2d.calibrate()


plt.figure()
plt.matshow(sweetSpot)
plt.title('Sweet Spot')
plt.colorbar()

plt.figure()
plt.matshow(t2d.CA,cmap=cm.gray)
plt.title('Inverse Filter')
plt.colorbar()

testnoise=0.03
s2n=addNoise2D(sig2,testnoise)
rec2=t2d.reconstruct(s2n)
plt.figure()
plt.matshow(rec2)
plt.colorbar()
plt.title('Recovered Sample - 6x6 block')
plt.figure()
plt.matshow(s2n)
plt.colorbar()

#rec1=t2d.reconstruct(addNoise2D(sig1,testnoise))
#plt.figure()
#plt.matshow(rec1,cmap=cm.gray)
#plt.title('Recovered sample-8x8 holes')
#plt.colorbar()
#
#plt.figure()
#plt.matshow(addNoise2D(sig1,testnoise))
#plt.title('8x8Signal')
#
#rec3=t2d.reconstruct(addNoise2D(sig3,testnoise))
#plt.figure()
#plt.matshow(rec3,cmap=cm.gray)
#plt.colorbar()
#plt.title('Recovered - ring, hole=10')
#
#rec4=t2d.reconstruct(addNoise2D(sig4,testnoise))
#plt.figure()
#plt.matshow(rec4)
#plt.colorbar()
#plt.title('Recovered - stripes 5px:11px')
#plt.figure()
#plt.matshow(addNoise2D(sig4,testnoise))
#plt.title('Signal+noise')