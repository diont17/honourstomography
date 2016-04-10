# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:58:27 2016

@author: Dion
"""

import numpy as np
import scipy
from scipy.optimize import minimize

class Tomography_1D(object):
    """
    """
    
    def __init__(self, sweetSpot):
        """
        
        Arguments:
        - `sweetSpot`:
        """
        self._sweetSpot = sweetSpot
        self.Amatrix= np.zeros([0,sweetSpot.shape[0]])
        self.b=np.zeros(0)
        return


    def calcSignal(self, smp):
        """
        
        Arguments:
        - `smp`:
        """
        
        M = smp.shape[0]
        N = self._sweetSpot.shape[0]

        lb=int(N/2)
        lf=+int(N/2)
        
        padSmp=np.zeros(M+N+N)        
        padSmp[N:N+M]=smp
        
        if M>N:
            signal = []
            for i in xrange(lb,M+N+lb):
                signal.append(sum(self._sweetSpot*padSmp[i-lb:i+lf]))

            return np.array(signal)

        else:
            print "Sample too small - sweet spot size: %d"%N
            return None
            
    def addTrainingData(self,pattern,signal):
        N = self._sweetSpot.shape[0]
        lb=int(N/2)
        S=signal.shape[0]
        B=pattern.shape[0]
        
        newA=np.zeros([B-N,N])
        newb=pattern[lb:B-lb]
        
        padSig=np.zeros(S+N+N)        
        padSig[N:N+S]=signal[:]
        
        for i in xrange(lb,B-N+lb,1):
            newA[i-lb,:] = padSig[i+N:i+N+N]
            
        self.b=np.concatenate((self.b,newb),axis=0)
        self.Amatrix=np.concatenate((self.Amatrix,newA),axis=0)
        
        self.calibrate()
        return
        
    def lossfn(self,CAx):
        out=np.dot(self.Amatrix,CAx)
        return np.sum(np.abs(out-self.b)) + 16000 * np.std(CAx)
#        return np.sum(np.abs(out-self.b)) + 80*np.sum(np.abs(np.diff(CAx)))
        
        
    def calibrate(self):

        # Solve Ax = b
        N = self._sweetSpot.shape[0]
        lb=int(N/2)
        S=self.Amatrix.shape[0]
        B=self.b.shape[0]

        self.CA0,_,_,_ = np.linalg.lstsq(self.Amatrix,self.b)
#        filt=exp(-0.5*((arange(0,N)-(0.5*N))/(0.2*N))**2)
#        self.weightedCA=self.CA*filt
#        self.weightedCA*=sum(self.CA)/sum(self.weightedCA)

        self.optres = minimize(self.lossfn, x0=np.ones(N))
        self.CA=self.optres.x        
  
        
    def reconstruct(self, signal):

        N = self._sweetSpot.shape[0]
        S = signal.shape[0]
        rec = np.zeros(S+N)
        
        lb=int(N)/2        
                
        padSig=np.zeros(N+S+N)        
        padSig[0:S]=signal[:]
        
        for i in xrange(0,S):
            rec[i] = np.sum(padSig[i:i+N]*self.CA)
        return rec[0:S]