# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:02:34 2016

@author: dion
"""


import numpy as np
import scipy.ndimage as img
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import cm

from mpl_toolkits.axes_grid1 import make_axes_locatable

import TwoDTomography as tm


def addNoise2D(arr, noisepercent):
    arrx= arr.shape[1]
    arry= arr.shape[0]
    noise = noisepercent - 2*noisepercent*np.max(arr)*np.random.rand(arry,arrx)
    return arr+noise

def makeSweetSpot(std):
    size=20
    x, y = np.float64(np.mgrid[-size/2 + 1:size/2 + 1, -size/2 + 1:size/2 + 1])
    x-=0.5
    y-=0.5
    #0.9 for narrow sweetspot
    #0.4 for wide sweetspot
    
    sweetSpot=np.exp(- ((x**2 + y**2)/ 2*std**2))
    sweetSpot*=(1.0/np.max(sweetSpot))
    return sweetSpot
    #sweetSpot=np.repeat(np.repeat(sweetSpot,2,axis=0),2,axis=1)
    #sweetSpot=np.ones((6,6))

t2ds=tm.Tomographer_2d(makeSweetSpot(0.2))#0.9))
t2dm=tm.Tomographer_2d(makeSweetSpot(0.2))#0.4))
t2dl=tm.Tomographer_2d(makeSweetSpot(0.2))


np.random.seed(2)
diagholes4= np.array([[0,0,0,0,2,2,2,2,2,2,2,2],[0,0,0,0,2,2,2,2,2,2,2,2],[0,0,0,0,2,2,2,2,2,2,2,2],[0,0,0,0,2,2,2,2,2,2,2,2],
                [1,1,1,1,0,0,0,0,1,1,1,1],[1,1,1,1,0,0,0,0,1,1,1,1],[1,1,1,1,0,0,0,0,1,1,1,1],[1,1,1,1,0,0,0,0,1,1,1,1],
                [1,1,1,1,1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1,0,0,0,0]])
diagholes12=np.repeat(np.repeat(diagholes4,3,axis=0),3,axis=1)

square6=np.array([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])
square12=np.repeat(np.repeat(square6,2,axis=0),2,axis=1)

ring12=img.imread('whitering.png', flatten=True)[::5,::5]
phan=img.imread('sheplog.png', flatten=True) [::1,::1]
ring12/=255.0
square12=ring12

#%%Training
trainingnoise=0.001
t2ds.addTrainingData(square12,addNoise2D(t2ds.calcSignal(square12),trainingnoise))
t2dm.addTrainingData(square12,addNoise2D(t2dm.calcSignal(square12),trainingnoise))
t2dl.addTrainingData(square12,addNoise2D(t2dl.calcSignal(square12),trainingnoise))

t2ds.lam=0
t2dm.lam=5
t2dl.lam=10

t2ds.calibrate()
t2dm.calibrate()
t2dl.calibrate()

#%%Reconstruct
noise=0.001
ssdh4 = addNoise2D(t2ds.calcSignal(diagholes4),noise)
ssdh12= addNoise2D(t2ds.calcSignal(diagholes12),noise)
ssrng12= addNoise2D(t2ds.calcSignal(ring12),noise)
ssphan=addNoise2D(t2ds.calcSignal(phan),noise)

srdh4 = t2ds.reconstruct(ssdh4)
srdh12= t2ds.reconstruct(ssdh12)
srrng12= t2ds.reconstruct(ssrng12)
srphan=t2ds.reconstruct(ssphan)

msdh4 = addNoise2D(t2dm.calcSignal(diagholes4),noise)
msdh12= addNoise2D(t2dm.calcSignal(diagholes12),noise)
msrng12= addNoise2D(t2dm.calcSignal(ring12),noise)
msphan= addNoise2D(t2dm.calcSignal(phan),noise)

mrdh4 = t2dm.reconstruct(msdh4)
mrdh12= t2dm.reconstruct(msdh12)
mrrng12= t2dm.reconstruct(msrng12)
mrphan= t2dm.reconstruct(msphan)

lsdh4 = addNoise2D(t2dl.calcSignal(diagholes4),noise)
lsdh12= addNoise2D(t2dl.calcSignal(diagholes12),noise)
lsrng12= addNoise2D(t2dl.calcSignal(ring12),noise)
lsphan=addNoise2D(t2dl.calcSignal(phan),noise)

lrdh4 = t2dl.reconstruct(lsdh4)
lrdh12= t2dl.reconstruct(lsdh12)
lrrng12= t2dl.reconstruct(lsrng12)
lrphan= t2dl.reconstruct(lsphan)

fig1,(ax11,ax12,ax13)=plt.subplots(nrows=1,ncols=3,figsize=(15,5))
ms11   = ax11.matshow(t2ds.sweetSpot)
ax11.set_title('Sweet Spot -small')
ms12   = ax12.matshow(t2dm.sweetSpot)
ax12.set_title('Sweet Spot -med')
ms13=ax13.matshow(t2dl.sweetSpot)
ax13.set_title('SweetSpot-large')
div13  = make_axes_locatable(ax13)
cax13  = div13.append_axes("right", size="20%", pad=0.05)
cbar13 = plt.colorbar(ms11)
cax13.set_visible(False)

fig2,(ax21,ax22,ax23)=plt.subplots(nrows=1,ncols=3,figsize=(15,5))
ms21=ax21.matshow(srdh4)
ax21.set_title('R-diaghole4-small')
ms22=ax22.matshow(mrdh4)
ax22.set_title('R-Diaghole4-med')
ms23=ax23.matshow(lrdh4)
ax23.set_title('R-Diaghole4-large')
plt.colorbar(ms21)

fig3,((ax31,ax32,ax33),(ax31b,ax32b,ax33b))=plt.subplots(nrows=2,ncols=3,figsize=(15,10))
ms31=ax31.matshow(srdh12)
ax31.set_title('R-diaghole12-small')
ms32=ax32.matshow(mrdh12)
ax32.set_title('R-Diaghole12-med')
ms33=ax33.matshow(lrdh12)
ax33.set_title('R-Diaghole12-large')
plt.colorbar(ms31)
ax31b.matshow(ssdh12)
ax31b.set_title('S-diaghole12-small')
ax32b.matshow(msdh12)
ax32b.set_title('S-diaghole12-med')
ax33b.matshow(lsdh12)
ax33b.set_title('S-diaghole12-large')

fig4,((ax41,ax42,ax43),(ax41b,ax42b,ax43b))=plt.subplots(nrows=2,ncols=3,figsize=(15,5))
ms41=ax41.matshow(srrng12)
ax41.set_title('R-ring12-small')
ms42=ax42.matshow(mrrng12)
ax42.set_title('R-ring12-med')
ms43=ax43.matshow(lrrng12)
ax43.set_title('R-Ring12-large')
plt.colorbar(ms41)
ax41b.matshow(ssrng12)
ax42b.matshow(msrng12)
ax43b.matshow(lsrng12)

fig5,ax5=plt.subplots(nrows=2,ncols=3,figsize=(15,5))
ms51=ax5[0,0].matshow(srphan)
ax5[0,0].set_title('R-phantom-small')
ms52=ax5[0,1].matshow(mrphan)
ax5[0,1].set_title('R-phantom-med')
ms53=ax5[0,2].matshow(lrphan)
ax5[0,2].set_title('R-phantom-large')
plt.colorbar(ms51)
ax5[1,0].matshow(ssphan)
ax5[1,1].matshow(msphan)
ax5[1,2].matshow(lsphan)
