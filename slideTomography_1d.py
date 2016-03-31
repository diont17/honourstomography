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
            for i in range(lb,M+lb+lf):
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
            S=signal.shape[0]
#            A = zeros([signal.shape[0]-N,N])
            A=zeros([S-N,N])
#            b = signal[:-N] - pattern[N/2:-N/2-N]
            b=pattern[0:S-N]
            
            padSig=np.zeros(lb+S+lf)        
            padSig[lb:lb+S]=signal[:]

            for i in range(lb,S-N):
                A[i,:] = padSig[i-lb:i+lf]

            self.Amatrix=A
            self.b=b
            self.CA,self.coeff_resid,self.coeff_rank,self.coeff_s = lstsq(A,b)

        
        

    def reconstruct(self, signal):
        """
        
        Arguments:
        - `signal`:
        """

        N = self._sweetSpot.shape[0]
        S = signal.shape[0]
        rec = zeros(S+100)
        
        lb=int(N)/2        
        lf=int(N)/2
        
        padSig=np.zeros(lb+S+lf)        
        padSig[lb:lb+S]=signal[:]

        
        for i in range(lb,S):
            rec[i] = sum(padSig[i-lb:i+lf]*self.CA)
        return rec

     