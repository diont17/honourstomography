# -*- coding: utf-8 -*-
"""
Find optimum drec array
Created on Fri Mar 11 14:26:41 2016

@author: dion
"""

import numpy as np
import random
import atexit


sampleamp=3*np.array([0,0,0,0,0,0,5,5,5,0,0,0,0,0,5,5,5,0,0,0,0,0,5,5,5,0,0,0,0,0,5,5,5,0,0,0,0,0,5,5,5,0,0,0,0,0,5,5,5,0,0,0,0,0,5,5,5,0,0,0,0,0,5,5,5,5,0,0,0,0])
#sampleamp=np.array([0,0,0,0,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,5,5,5,5,5,5,5,8,8,8,6,6,6,6,6,6,6,4,4,4,5,5,5,5,3,3,3,3,3,1,1,1,1,1,0,0,0,0])

windowlength=3
#window=np.array([0.3,0.7,1,0.7,0.3])
window=np.array([1,1,1,1,1])
rightstop=len(sampleamp)-windowlength
signal=np.zeros(len(sampleamp))

random.seed(7)
#move window through sampleamp
for i in range(3,rightstop):
    spot=sampleamp[i-2:i+3]
    spot=spot*window
    spotsignal=np.sum(spot)
    signal[i]=spotsignal+(3-6*random.random())

#Recover
rec=np.zeros(len(sampleamp))
prevwindow=np.zeros(windowlength)
drec=np.zeros(len(sampleamp))

#exit nicely for long calcs:
def report():
    print "Exiting.."
    print "Best Score: {:6.2f} from dcoff:".format(bestscore)
    print bestcoff
atexit.register(report)

#Find optimized dcoff

bestcoff=[0,0,0,0,0]
bestscore=1000000
halfrange=140
done=0

for a in xrange(-50,50):
    for b in xrange(-50,50):
        for c in xrange(-100,100):
            dcoff=np.array([a,b,c])*-0.01
            
            for i in range(3,rightstop):
#                drec[i]=np.sum(dcoff*signal[i-2:i+3])
#                rec[i]=1.0*signal[i]-1.0*drec[i]
                #this might be faster
                rec[i]=signal[i] + dcoff[0]*signal[i-2] +dcoff[1]*signal[i-1] + dcoff[2]*signal[i] + dcoff[1]*signal[i+1] + dcoff[0]*signal[i+2]

            score=np.sum((sampleamp-rec)**2)
            print "{:7d}: abc ({:+4.0f},{:+4.0f},{:+4.0f}) s:{:8.2f} b:{:6.2f}".format(done,a,b,c,score,bestscore)
            if score<bestscore:
                bestscore=score
                bestcoff=dcoff*-10
            done+=1

print "\nSearch completed\n"