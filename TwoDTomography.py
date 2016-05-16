# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:02:26 2016

@author: dion
"""

import numpy as np
import scipy.ndimage as img
from scipy.optimize import minimize

class Tomographer_2d(object):
    
    def __init__(self,sweetSpot):
        self.sweetSpot= sweetSpot
        self.Nx= sweetSpot.shape[1]
        self.Ny= sweetSpot.shape[0]
        
        self.Amatrix=np.zeros([0,self.Nx*self.Ny])
        self.b=np.zeros(0)
        self.CA=np.zeros_like(sweetSpot)
        self.lam=0
    
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
#            sig[m, -11:-1]=sig[m,-12]
        return sig
    
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
        padsig[lby:Sy+lby, lbx:Sx+lbx]=sig
        segment=np.zeros((self.Ny,self.Nx))
        
        for m in xrange(0,Sy):
            for n in xrange(0,Sx):
                padm=m+lby
                padn=n+lbx
                segment[0:self.Nx:,0:self.Ny] = padsig[padm-lby:padm+lby, padn-lbx:padn+lbx]
                
                newA[m*Sy+n,:]  = segment.reshape(-1)
                newb[(m*Sx) +n] = padSmp[m,n]
            
        
        self.Amatrix=np.concatenate((self.Amatrix,newA[:]),axis=0)
        self.b=np.concatenate((self.b,newb[:]),axis=0)


    def calibrate(self):
        
        self.Amatrix+=(self.lam**2) *np.eye(N=self.Amatrix.shape[0],M=self.Amatrix.shape[1],k=0)
        self.CA0,_,_,_= np.linalg.lstsq(self.Amatrix,self.b)       
#        self.optres=minimize(self.lossfn,x0=0.3*np.ones(self.Nx*self.Ny))
#        self.CA=self.optres.x.reshape((self.Ny,self.Nx))
        self.CA=self.CA0.reshape((self.Ny,self.Nx))
        
    def lossfn(self,CAx):
        return 1e-3*np.sum((np.dot(self.Amatrix,CAx)-self.b)**2) + 1e-3*(self.lam) * np.sum((CAx)**2)
#        return np.sum(np.abs(out-self.b)) + 80*np.sum(np.abs(np.diff(CAx)))
                
        
    def reconstruct(self,sig):
        rec=np.zeros_like(sig)
        rec=img.convolve(sig,weights=self.CA,mode='constant', cval=0)
        return rec

