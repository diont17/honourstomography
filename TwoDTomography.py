# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:02:26 2016

@author: dion
"""

import numpy as np
import scipy.ndimage as img

import matplotlib.pylab as plt

class Tomographer_2d(object):
    
    def __init__(self,sweetSpot):
        self.sweetSpot= sweetSpot
        self.Nx= sweetSpot.shape[1]
        self.Ny= sweetSpot.shape[0]
        
        self.Amatrix=np.zeros([0,sweetSpot.size])
        self.b=np.zeros(0)
        self.CA=np.zeros_like(sweetSpot)
    
    def calcSignal(self, smp):
        Ox= smp.shape[1]
        Oy= smp.shape[0]
        
        sig= np.zeros_like(smp)
        sig=img.convolve(smp, weights=self.sweetSpot, mode='constant', cval=0)    
        
#        for i in xrange(Oy-self.Ny):
#            for j in xrange(Ox-self.Nx):
#                
#                zone=smp[i:i+self.Ny, j:j+self.Nx]
#                sig[i,j] = np.sum(self.sweetSpot*zone)
        return sig
    
    def addTrainingData(self, smp, sig):
        lbx=int(self.Nx/2)
        lby=int(self.Ny/2)
        
        Sx=sig.shape[1]
        Sy=sig.shape[0]
        
        Ox=smp.shape[1]
        Oy=smp.shape[0]        

        newA=np.zeros([Sx*Sy,Ox*Oy])
        newb=np.zeros([Ox*Oy])
        
        #pad sample so looking back works                
        padsig=np.zeros([self.Ny + Sy, self.Nx+Sx])
        padsig[lby:lby+Sy, lbx:lbx+Sx]=sig
        segment=np.zeros((self.Ny,self.Nx))
        
        for m in xrange(0,self.Ny):
            for n in xrange(0,self.Nx):
                padm=m+lby
                padn=n+lbx
                segment[0:50,0:50]=padsig[padm-lby:padm+lby, padn-lbx:padn+lbx]
                
                newA[m*self.Ny+n][:]=segment.reshape(-1)
                newb[(m*self.Nx) +n]=smp[m,n]
            
        
        self.Amatrix=newA
        self.b=newb


    def calibrate(self):
        self.CA,_,_,_= np.linalg.lstsq(self.Amatrix,self.b)
        self.CA=self.CA.reshape((self.Ny,self.Nx))
        
    def reconstruct(self,sig):
        rec=np.zeros_like(sig)
        rec=img.convolve(sig,self.CA,mode='constant', cval=0)
        return rec
        
size=50
x, y = np.mgrid[-size/2 + 1:size/2 + 1, -size/2 + 1:size/2 + 1]

sweetSpot=np.exp(- ((x**2 + y**2)/ 2*0.01**2)) / 2448.688
t2d=Tomographer_2d(sweetSpot)

#smp=img.imread('whitesquare.png',flatten=True)[::8,::8]
smp=img.imread('whitering.png', flatten=True)[::8,::8]
sig=t2d.calcSignal(smp)

plt.imshow(sig)
plt.figure()
t2d.addTrainingData(smp,sig)
plt.imshow(t2d.Amatrix)
t2d.calibrate()
plt.imshow(t2d.CA)

rec=t2d.reconstruct(sig)
plt.imshow(rec)