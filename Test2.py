
# -*- coding: utf-8 -*-
"""

Created on Mon Mar 14 17:04:30 2016

@author: Dion
"""


import numpy as np
import scipy as sci
import scipy.signal as sig
import matplotlib.pylab as plt

#Test samples, all length=60
flatsample=3*np.array([0,0,0,0,0,0,0,0,0,0,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,0,0,0,0])
slopsample=0.5*np.array([0,0,0,0,0,0,0.5,0.5,1,1,1.5,2,3,4,5,6,8,10,12,14,16,18,20,23,26,29,32,32,32,32,32,33,34,34,34,33,32,32,31,
                   30,29,28,27,26,25,24,23,22,21,20,18,16,14,12,10,8,6,4,2,0])
sinesample=np.zeros(60)
sinesample[3:57]=10+5*np.sin(np.linspace(0,3*np.pi,54))
stepsample=3*np.array([0,0,0,0,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,5,5,5,5,5,5,5,8,8,8,6,6,6,6,6,6,6,4,4,4,5,5,5,5,3,3,3,3,3,1,1,1,1,1,0,0,0,0])
gaussample=15*sig.gaussian(60,10)
rounsample=0.7*np.array([0,0,0,0,0.5,0.6,0.8,1,1.5,2,3,4,5,7,9,12,15,18,19,20,21,21.5,22,22.5,23,23.2,23.1,23.1,23.2,23.3,23.2,23.1,28,23.2,23.2,22.5,22,21.5,
                     21,20.5,20,19,17,15,12,9,7,5,4,3,2,1.5,1,1,0.8,0.4,0,0,0,0])

window=np.array([0.3,0.7,1,0.7, 0.3])
#window=np.array([1,1,1,1,1])
rightstop=57

#dcoff arrays
#round window=[0.3,0.7,1,0.7, 0.3]
#-6:6, 0.1s=13.2076
#dcoff=np.array([0.4,0,5.9,0,0.4])

#dcoff=np.array([0.3,-1.1,8.3,-1.1,0.3])
#dcoff=np.array([1,-1.4,7.5,-1.4,1])
#dcoff=np.array([1.6,-1,5.6,-1,1.6])
#dcoff=np.array([1.9,-2.3,7.5,-2.3,1.9])
dcoff=np.array([1.5,-2,7.7,-2,1.5])

#For flat window=[1,1,1,1,1]
#s=17.37
#dcoff=np.array([1.3,-1.3,8,-1.3,1.3])
#s=11.6
#dcoff=np.array([1.6,-3.4,11.6,-3.4,1.6])

dcoff=dcoff*0.1

np.random.seed(651654897)
#storednoise=np.zeros(60)
storednoise=1.5-3*np.random.rand(60)

def makesignal(orig):
    signal=np.zeros(len(orig))
    #Moves through the sample and sums the sample values in the sweet spot to generate the signal
    #Adds noise for each measurement
    for i in range(3,rightstop):
        spot=orig[i-2:i+3]
        spot=spot*window
        spotsignal=np.sum(spot)
        signal[i]=spotsignal+(storednoise[i])
    return signal

def recover(signal):
    rec=np.zeros(len(signal))
    drec=np.zeros(len(signal))
    
    #Calculates a correction to the signal drec by stepping through the signal and taking a weighted sum of the values around it
    #Then the corrected signal to the signal to generate the recovered sample rec
    for i in range(3,rightstop):
        drec[i]=np.sum(dcoff*signal[i-2:i+3])
        rec[i]=1.0*signal[i]-1.0*drec[i]
    return rec
    
#Test with flat sample
flatrec=recover(makesignal(flatsample))
plt.figure()
plt.subplot(2,3,1)
plt.plot(flatsample,label='o')
plt.plot(flatsample+storednoise,label='o+n')
plt.plot(flatrec,label='rec')
plt.grid()
plt.title('Flat sample')
print dcoff
print "flat score {:g}".format(np.sum((flatsample-flatrec)**2))


#Test with changing slope sample
sloprec=recover(makesignal(slopsample))

plt.subplot(2,3,2)
plt.plot(slopsample,label='o')
plt.plot(slopsample+storednoise,label='o+n')
plt.plot(sloprec,label='rec')
plt.grid()
plt.title('sloped sample')

print "slope score {:g}".format(np.sum((slopsample-sloprec)**2))


#Sine sample
sinerec=recover(makesignal(sinesample))
plt.subplot(2,3,3)
plt.plot(sinesample,label='o')
plt.plot(sinesample+storednoise,label='o+n')
plt.plot(sinerec,label='rec')
plt.grid()
plt.title('sine sample')
print "sine score {:g}".format(np.sum((sinesample-sinerec)**2))

#Stepped sample
steprec=recover(makesignal(stepsample))

plt.subplot(2,3,4)
plt.plot(stepsample,label='o')
plt.plot(stepsample+storednoise,label='o+n')
plt.plot(steprec,label='rec')
plt.grid()
plt.title('stepped sample')
#plt.text(0,10,str(dcoff))
print "stepped score {:g}".format(np.sum((stepsample-steprec)**2))

#gaussian sample
gausrec=recover(makesignal(gaussample))

plt.subplot(2,3,5)
plt.plot(gaussample,label='o')
plt.plot(gaussample+storednoise,label='o+n')
plt.plot(gausrec,label='rec')
plt.grid()
plt.title('gaussian sample')
print "gauss score {:g}".format(np.sum((gaussample-gausrec)**2))

#round sample
rounrec=recover(makesignal(rounsample))

plt.subplot(2,3,6)
plt.plot(rounsample,label='o')
plt.plot(rounsample+storednoise,label='o+n')
plt.plot(rounrec,label='rec')
plt.grid()
plt.legend()
plt.title('round sample')
print "round score {:g}".format(np.sum((rounsample-rounrec)**2))
