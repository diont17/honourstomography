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

        if M>N:
            signal = []
            for i in range(M-N):
                signal.append(sum(self._sweetSpot*smp[i:i+N]))

            return array(signal)

        else:
            print "Sample too small - sweet spot size: %d"%N
            return None
            
        
    def calibrate(self, pattern):
        """
        
        Arguments:
        - `pattern`:
        """
        
        signal = self.calcSignal(pattern)

        if isinstance(signal,ndarray):
        
            # Solve Ax = b

            N = self._sweetSpot.shape[0]


            A = zeros([signal.shape[0]-N,N])
            b = signal[:-N] - pattern[N/2:-N/2-N]

            for i in range(signal.shape[0]-N):
                A[i,:] = signal[i:i+N]


            self._coeffArray,resid,rank,s = lstsq(A,b)

        
        

    def reconstruct(self, signal):
        """
        
        Arguments:
        - `signal`:
        """

        N = self._sweetSpot.shape[0]
        rec = zeros([signal.shape[0]-N])
        
        for i in range(signal.shape[0]-N):
            rec[i] = signal[i]-sum(signal[i:i+N]*self._coeffArray)
    

        return rec
