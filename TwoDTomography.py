# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:02:26 2016

@author: dion
"""

import numpy as np
import scipy.ndimage as img
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import cm

class Tomographer_2d(object):
    
    def __init__(self,sweetSpot):
        self.sweetSpot= sweetSpot
        self.Nx= sweetSpot.shape[1]
        self.Ny= sweetSpot.shape[0]
        
        self.Amatrix=np.zeros([0,self.Nx*self.Ny])
        self.b=np.zeros(0)
        self.CA=np.zeros_like(sweetSpot)
        self.lam=6000
    
    def calcSignal(self, smp):
        Ox= smp.shape[1]
        Oy= smp.shape[0]
        lbx=int(self.Nx/2)
        lby=int(self.Ny/2)
        
        sig= np.zeros((Oy+self.Ny, Ox+self.Nx))
#        sig=img.convolve(smp, weights=self.sweetSpot, mode='constant', cval=0)    
#        
#        for i in xrange(0, Oy-self.Ny):
#            for j in xrange(Ox-self.Nx):
#                
#                zone=smp[i:i+self.Ny, j:j+self.Nx]
#                sig[i,j] = np.sum(self.sweetSpot*zone)
#                
        padsmp=np.zeros([2*self.Ny + Oy, 2*self.Nx + Ox])
        padsmp[self.Ny:self.Ny+Oy, self.Nx:self.Nx+Ox]=smp
                
        for m in xrange(0,Oy+self.Ny):
            for n in xrange(0,Ox+self.Nx):
                padm=m+lby
                padn=n+lbx
                segment=padsmp[padm-lby:padm+lby, padn-lbx:padn+lbx]
                sig[m,n]=np.sum(segment*self.sweetSpot)

        return sig,padsmp
    
    def addTrainingData(self, smp, sig):
        lbx=int(self.Nx/2)
        lby=int(self.Ny/2)
        
        Sx=sig.shape[1]
        Sy=sig.shape[0]
        
        Ox=smp.shape[1]
        Oy=smp.shape[0]        

        newA=np.zeros([Sx*Sy,self.Nx*self.Ny])
        newb=np.zeros([Sx*Sy])
        
        padSmp = np.zeros([Sy,Sx])
        padSmp[Sy/2-Oy/2:Sy/2+Oy/2,Sx/2-Ox/2:Sy/2+Oy/2] = smp
        
        #pad sample so looking back works                
        padsig=np.zeros([self.Ny + Sy, self.Nx+Sx])
#        padsig[lby:lby+Sy, lbx:lbx+Sx]=sig
        padsig[0:Sy, 0:Sx]=sig
        segment=np.zeros((self.Ny,self.Nx))
        
        for m in xrange(0,Sy):
            for n in xrange(0,Sx):
                padm=m+lby
                padn=n+lbx
                segment[0:self.Nx:,0:self.Ny] = padsig[padm-lby:padm+lby, padn-lbx:padn+lbx]
                
                newA[m*Sy+n,:]  = segment.reshape(-1)
                newb[(m*Sx) +n] = padSmp[m,n]
            
        
        self.Amatrix=np.concatenate((self.Amatrix,newA),axis=0)
        self.b=np.concatenate((self.b,newb),axis=0)


    def calibrate(self):
#        self.CA,_,_,_= np.linalg.lstsq(self.Amatrix,self.b)
#        self.CA=self.CA.reshape((self.Ny,self.Nx))
        self.optres=minimize(self.lossfn,x0=np.ones(self.Nx*self.Ny))
        self.CA=self.optres.x.reshape((self.Ny,self.Nx))
        
    def lossfn(self,CAx):
        return np.sum((np.dot(self.Amatrix,CAx)-self.b)**2) + (self.lam) * np.sum((CAx)**2)
#        return np.sum(np.abs(out-self.b)) + 80*np.sum(np.abs(np.diff(CAx)))
                
        
    def reconstruct(self,sig):
        rec=np.zeros_like(sig)
        rec=img.convolve(sig,weights=self.CA,mode='constant', cval=0)
        return rec

def addNoise2D(arr, noisepercent):
    arrx= arr.shape[1]
    arry= arr.shape[0]
    noise = noisepercent - 2*noisepercent*np.max(arr)*np.random.rand(arry,arrx)
    return arr+noise

np.random.seed(2)
size=10
x, y = np.float64(np.mgrid[-size/2 + 1:size/2 + 1, -size/2 + 1:size/2 + 1])
x-=0.5
y-=0.5
#0.9 for narrow sweetspot
#0.4 for wide sweetspot
std=0.4

sweetSpot=np.exp(- ((x**2 + y**2)/ 2*std**2))
sweetSpot*=(1.0/np.max(sweetSpot))
#sweetSpot=np.ones((6,6))
t2d=Tomographer_2d(sweetSpot)

#smp=img.imread('whitesquare.png',flatten=True)[::80,::80]
smp1= np.array([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,0,0,1,1],[1,1,0,0,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])
smp1= np.array([[0,0,0,0,1,1,1,1,1,1,1,1],[0,0,0,0,1,1,1,1,1,1,1,1],[0,0,0,0,1,1,1,1,1,1,1,1],[0,0,0,0,1,1,1,1,1,1,1,1],
                [1,1,1,1,0,0,0,0,1,1,1,1],[1,1,1,1,0,0,0,0,1,1,1,1],[1,1,1,1,0,0,0,0,1,1,1,1],[1,1,1,1,0,0,0,0,1,1,1,1],
                [1,1,1,1,1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1,0,0,0,0]])

smp2=np.array([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])
smp3=img.imread('whitering.png', flatten=True)[::10,::10]
smp4=img.imread('stripes.png', flatten=True) [::10,::2]
smp3/=255.0
smp4/=255.0
smp1=np.repeat(np.repeat(smp1,4,axis=0),4,axis=1)

sig1,padsmp1=t2d.calcSignal(smp1)
sig2,_=t2d.calcSignal(smp2)
sig3,_=t2d.calcSignal(smp3)
sig4,_=t2d.calcSignal(smp4)

#plt.matshow(padsmp)
#plt.matshow(smp,cmap=cm.gray)
#plt.title('sample')
#plt.figure()
#plt.matshow(sig,cmap=cm.gray)
#plt.title('signal')

#t2d.addTrainingData(smp1,addNoise2D(sig1,0.05))
#t2d.addTrainingData(smp1,addNoise2D(sig1,0.11))
#t2d.addTrainingData(smp1,addNoise2D(sig1,0.21))

t2d.addTrainingData(smp2,addNoise2D(sig2,0.01))
t2d.addTrainingData(smp2,addNoise2D(sig2,0.01))
t2d.addTrainingData(smp2,addNoise2D(sig2,0.01))
t2d.lam=16000
t2d.calibrate()

plt.figure()
plt.matshow(sweetSpot)
plt.title('Sweet Spot')
plt.colorbar()

plt.figure()
plt.matshow(t2d.CA,cmap=cm.gray)
plt.title('Inverse Filter')
plt.colorbar()
testnoise=0.4
rec2=t2d.reconstruct(addNoise2D(sig2,testnoise))
#rec2+=(-1.0*rec2.min())
plt.figure()
plt.matshow(rec2,cmap=cm.gray)
plt.title('Recovered Sample - 6x6 block')
plt.colorbar()

rec1=t2d.reconstruct(addNoise2D(sig1,testnoise))
plt.figure()
plt.matshow(rec1,cmap=cm.gray)
plt.title('Recovered sample- hole=4')
plt.colorbar()

rec3=t2d.reconstruct(addNoise2D(sig3,testnoise))
plt.figure()
plt.matshow(rec3,cmap=cm.gray)
plt.colorbar()
plt.title('Recovered - ring, hole=10')

rec4=t2d.reconstruct(addNoise2D(sig4,testnoise))
plt.figure()
plt.matshow(rec4)
plt.colorbar()
plt.title('Recovered - stripes 5px:11px')