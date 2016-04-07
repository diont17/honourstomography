# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:58:27 2016

@author: Dion
"""

from pylab import *


class Tomography_1D(object):
    """
    """
    
    def __init__(self, sweetSpot):
        """
        
        Arguments:
        - `sweetSpot`:
        """
        self._sweetSpot = sweetSpot
        self.Amatrix=zeros([0,sweetSpot.shape[0]])
        self.b=zeros(0)


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

            return array(signal)

        else:
            print "Sample too small - sweet spot size: %d"%N
            return None
            
    def addTrainingData(self,pattern,signal):
        N = self._sweetSpot.shape[0]
        lb=int(N/2)
        S=signal.shape[0]
        B=pattern.shape[0]
        
        newA=zeros([B-N,N])
        newb=pattern[lb:B-lb]
        
        padSig=np.zeros(S+N+N)        
        padSig[N:N+S]=signal[:]
        
        for i in xrange(lb,B-N+lb,1):
            newA[i-lb,:] = padSig[i+N:i+N+N]
            
        self.b=concatenate((self.b,newb),axis=0)
        self.Amatrix=concatenate((self.Amatrix,newA),axis=0)
        
        self.calibrate()
        return
        
    def calibrate(self):

        # Solve Ax = b
        N = self._sweetSpot.shape[0]
        lb=int(N/2)
        S=self.Amatrix.shape[0]
        B=self.b.shape[0]

        self.CA,self.coeff_resid,self.coeff_rank,self.coeff_s = lstsq(self.Amatrix,self.b)
        filt=exp(-0.5*((arange(0,N)-(0.5*N))/(0.2*N))**2)
        self.weightedCA=self.CA*filt
        self.weightedCA*=sum(self.CA)/sum(self.weightedCA)
        return
        
    def reconstruct(self, signal):

        N = self._sweetSpot.shape[0]
        S = signal.shape[0]
        rec = zeros(S+N)
        
        lb=int(N)/2        
                
        padSig=np.zeros(N+S+N)        
        padSig[0:S]=signal[:]
        
        for i in xrange(0,S):
            rec[i] = sum(padSig[i:i+N]*self.CA)
        return rec[0:S]