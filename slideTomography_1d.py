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
            
            
    def calibrate(self, pattern, signal):
        """
        
        Arguments:
        - `pattern`:
        """
        lb=self._sweetSpot.shape[0]/2
        lf=self._sweetSpot.shape[0]/2
                
        if isinstance(signal,ndarray):
        
            # Solve Ax = b
            
            N = self._sweetSpot.shape[0]
            lb=int(N/2)
            S=signal.shape[0]
            B=pattern.shape[0]
#            A = zeros([signal.shape[0]-N,N])
#            A=zeros([S-N,N])
            A=zeros([B-N,N])
#            b = signal[:-N] - pattern[N/2:-N/2-N]
            b=pattern[lb:B-lb]
            self.Amatrix=A
            self.b=b           
            
            padSig=np.zeros(S+N+N)        
            padSig[N:N+S]=signal[:]
            
            for i in xrange(lb,B-N+lb,1):
                A[i-lb,:] = padSig[i+N:i+N+N]


            self.CA,self.coeff_resid,self.coeff_rank,self.coeff_s = lstsq(A,b)
            filt=exp(-0.5*((arange(0,N)-(0.5*N))/(0.2*N))**2)
#            filt=np.arange(N)
            self.weightedCA=self.CA*filt
            self.weightedCA*=sum(self.CA)/sum(self.weightedCA)
        
    def reconstruct(self, signal):
        """
        
        Arguments:
        - `signal`:
        """

        N = self._sweetSpot.shape[0]
        S = signal.shape[0]
        rec = zeros(S+N)
        
        lb=int(N)/2        
                
        padSig=np.zeros(N+S+N)        
        padSig[0:S]=signal[:]
        
        for i in xrange(0,S):
            rec[i] = sum(padSig[i:i+N]*self.CA)
        return rec[0:S]