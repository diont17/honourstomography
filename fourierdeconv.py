# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:52:12 2016

@author: dion
"""

#Fourrier

import numpy as np
import scipy.ndimage as img
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches
from skimage import restoration
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
#    sweetSpot=np.exp(- ((0.8*x**2 + 4*y**2)/ 2.0*std**2))
    sweetSpot=np.exp(- ((0.5*x**2 + 0.7*y**2)/ 1.0*std**2))*np.sin(0.05*np.pi*(x+y))**2
#
#    sweetSpot=np.float64([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,0,1,1,0],[0,0,0,1,0]])
#    sweetSpot=np.repeat(np.repeat(sweetSpot,4,axis=0),4,axis=1)
#    sweetSpot=img.gaussian_filter(sweetSpot,1.37,mode='constant')
   
    sweetSpot*=(1.0/np.max(sweetSpot))
    return sweetSpot

t2ds=tm.Tomographer_2d(makeSweetSpot(0.2))#0.9))

np.random.seed(2)
diagholes4= np.array([[0,0,0,0,2,2,2,2,2,2,2,2],[0,0,0,0,2,2,2,2,2,2,2,2],[0,0,0,0,2,2,2,2,2,2,2,2],[0,0,0,0,2,2,2,2,2,2,2,2],
                [1,1,1,1,0,0,0,0,1,1,1,1],[1,0,0,1,0,0,0,0,1,1,1,1],[1,0,0,1,0,0,0,0,1,1,1,1],[1,1,1,1,0,0,0,0,1,1,1,1],
                [1,1,1,1,1,1,1,0,0,0,0,0],[1,1,1,1,1,1,1,0,0,0,0,0],[1,1,1,1,1,1,1,0,0,0,0,0],[1,1,1,1,1,1,1,0,0,0,0,0]])
diagholes12=np.repeat(np.repeat(diagholes4,3,axis=0),3,axis=1)

square6=np.array([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])
square12=np.repeat(np.repeat(square6,2,axis=0),2,axis=1)

ring12=4*img.imread('whitering.png', flatten=True)[::8,::8]
phan=img.imread('sheplog.png', flatten=True) [::1,::1]
ring12/=255.0
phan/=255.0
#training sample
trainsmp=square12
#test sample
testsmp=ring12
#%%Training
trainingnoise=0.15
trainsig=addNoise2D(t2ds.calcSignal(trainsmp),trainingnoise)
t2ds.addTrainingData(trainsmp,trainsig)
#t2ds.addTrainingData(trainsmp,addNoise2D(t2ds.calcSignal(trainsmp),trainingnoise))

#%%Get psf by Fourier
FTS=fft.fftshift(fft.fft2(trainsig))

padsmp=np.zeros(trainsig.shape)
padsmp[10:-10,10:-10]=trainsmp

#padsmp[10:-10,-10:-1]=trainsmp[:,0:9]

FO=fft.fftshift(fft.fft2(padsmp))

FTSlev=np.percentile(np.abs(FTS),80)
FOlev=np.percentile(np.abs(FO),80)

threshFTS=np.where(np.abs(FTS)<FTSlev,0,FTS)
threshFO=np.where(np.abs(FO)<FOlev,0,FO)

threshFSS=np.where(np.abs(threshFO)>0,threshFTS/threshFO,0)#threshFTS)
iFSS=fft.ifftshift(fft.ifft2(threshFSS))

psfstart=(iFSS.shape[0]/2)-9
psf = np.abs(iFSS)[psfstart:psfstart+20,psfstart:psfstart+20]
psf=restoration.wiener(trainsig,trainsmp,balance=500)[psfstart:psfstart+20,psfstart:psfstart+20]
##%% Make psf2
#reTrain=restoration.wiener(trainsig,psf,balance=90)
#FRT=fft.fftshift(fft.fft2(reTrain))
#FRTlev=np.percentile(np.abs(FRT),90)
#threshFRT=np.where(np.abs(FRT)<FRTlev, 0, FRT)
#threshFSS2=np.where(np.abs(threshFO)>0, threshFRT/threshFO, threshFRT)
#iFSS2=fft.ifftshift(fft.ifft2(threshFSS2))
#plt.matshow(np.abs(iFSS2))
#plt.title('SweetSpot 2')
#FRT2=restoration.wiener(reTrain,np.abs(iFSS2)[14:19,14:19],balance=90)
#plt.matshow(reTrain)
#plt.title('reconstructed training sample')
#plt.matshow(FRT2)
#plt.title('rereconstructed training sample')
#%%Display fourier

fig3,((ax31,ax32,ax33),(ax31b,ax32b,ax33b),(ax31c,ax32c,ax33c))=plt.subplots(nrows=3,ncols=3,figsize=(15,10))
ms31=ax31.matshow(padsmp)
ax31.set_title('Training Sample')
ms32=ax32.matshow(trainsig)
ax32.set_title('Training Signal')
ms33=ax33.matshow(t2ds.sweetSpot)
ax33.set_title('SS')

ax31b.matshow(threshFO.real)
ax31b.set_title('FOreal')
ax32b.matshow(threshFTS.real)
ax32b.set_title('FSreal')
ax33b.matshow(threshFSS.real)
ax33b.set_title('FSS=FS/FO')

ax31c.matshow(np.abs(fft.ifft2(threshFO)))
ax31c.set_title('iFO')
ax32c.matshow(np.abs(fft.ifft2(threshFTS)))
ax32c.set_title('iFS')
ax33c.matshow(psf)#)np.abs(iFSS))
#ax33c.add_patch(matplotlib.patches.Rectangle((psfstart,psfstart),20,20,fill=None,edgecolor='r'))
ax33c.set_title('iFSS')

#%%Reconstruct
t2ds.lam=40
t2ds.calibrate()
noise=trainingnoise
ssdh12= addNoise2D(t2ds.calcSignal(testsmp),noise)
srdh12= t2ds.reconstruct(ssdh12)
Ddcnv=srdh12

#%% Deconvolve with found convolution
Sig = ssdh12
Sig+=Sig.min()
esf=Sig.max()
#psf=t2ds.sweetSpot
sigclip=2*(Sig/esf)
Rdcnv = restoration.richardson_lucy(sigclip,psf)
Wdcnv=restoration.wiener(sigclip,psf,balance=500,clip=False)
#Wdcnv+=1.0
Wdcnv*=esf/2
Rdcnv+=1
Rdcnv*=esf

#%%Division in fourier space
#Calculate F(testsignal)
FS=fft.fft2(Sig)
FSlev=np.percentile(np.abs(FS),80)
threshFS=np.where(np.abs(FS)<FSlev,0,FS)
#Calculate F(sweetspot),  (have to pad to correct size for signal)
padpsf=np.zeros_like(FS)
halfFS=FS.shape[0]/2
halfpsf=iFSS.shape[0]/2
padpsf[halfFS-halfpsf:halfFS+halfpsf,halfFS-halfpsf:halfFS+halfpsf]=iFSS

FSS=(fft.fft2(padpsf))
FSSlev=np.percentile(np.abs(FSS),80)
threshFSS=np.where(np.abs(FSS)<FSSlev,0,FS)

FR = np.where(np.abs(threshFSS)>0,threshFS/threshFSS,threshFS)
Fdcnv=np.abs(fft.ifft2(FR))

#%%Display
fig1,ax=plt.subplots(nrows=2,ncols=4,squeeze=False,figsize=(15,5))
ms11=ax[1,0].matshow(Ddcnv)
ax[1,0].set_title('Reconstruction')
fixclim=ms11.get_clim()

ms12=ax[1,1].matshow(Fdcnv)
#ms12.set_clim(fixclim)
ax[1,1].set_title('Division in fourier space')

ms13=ax[1,2].matshow(Rdcnv)
ax[1,2].set_title('RichardsonLucy Deconv')
#ms13.set_clim(fixclim)

ms14=ax[1,3].matshow(Wdcnv)
ax[1,3].set_title('Wiener Deconv')
#ms14.set_clim(fixclim)

ms02=ax[0,2].matshow(Sig)
ax[0,2].set_title('Signal')
ms01=ax[0,1].matshow(t2ds.sweetSpot)
ax[0,1].set_title('True sweetspot')
ms03=ax[0,3].matshow(psf)
ax[0,3].set_title('Constructed sweetspot')
ms00=ax[0,0].matshow(testsmp)
ax[0,0].set_title('Sample')
#div11  = make_axes_locatable(ax[0,0])
#cax11  = div11.append_axes("right", size="20%", pad=0.05)
#cbar11 = plt.colorbar(ms11)
#cax11.set_visible(False)
#div12  = make_axes_locatable(ax[0,1])
#cax12  = div12.append_axes("right", size="20%", pad=0.05)
#cbar12 = plt.colorbar(ms12)
#cax12.set_visible(False)
