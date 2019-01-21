#!/usr/bin/python3

import numpy as np
import sys

# p = exp(-alpha*t)
# q = exp(-beta*t)

import base_counting

def EM(tolerance,Nraw):
    #initialParam
    p,q,t = intialParam(Nraw)

    iteration = 0
    convergence = np.inf
    logLold = -np.inf

    while(convergence > tolerance):
        iteration += 1

        ProbM = K2P(p,q)
        N = Nraw/ProbM

        #E-step
        x_ = N[0][0]+N[1][1]+N[2][2]+N[3][3]        #diagonal of matrix (no change)
        y_ = x_ + N[0][2]+N[2][0]+N[1][3]+N[3][1]   #(no change) + transitions
        s1 = p*q*(x_)*0.25
        s2 = q*(1.0-p)*(y_)*0.125
        s3 = (1.0-q)*np.sum(N)*0.0625
        
        #M-step
        p = s1/(s1+s2)
        q = (s1+s2)/(s1+s2+s3)

        #calculation of R,t
        R,t = calcParam(p,q)
        logLnew = logLikelihood(p,q,Nraw)
        if(logLold > logLnew):
            print('log Likelihood error')
            break
        
        convergence = np.absolute(logLnew-logLold)
        print(iteration,'log-likelihood= ',logLnew,' R= ',R,' t= ',t)
        logLold = logLnew
    postprints(p,q)

def K2P(p,q):
    v = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            v[i][j] = (p*q*(1 if (i==j) else 0) \
                      + q*(1.0-p)*0.5*(1.0 if ((i%2)==(j%2)) else 0) \
                      + (1.0-q)*0.25)*0.25
    return v

def intialParam(N):
    #auxiliar constants
    P = (N[0][2]+N[1][3]+N[2][0]+N[3][1])/np.sum(N)
    Q = (N[0][1]+N[0][3]+N[1][0]+N[1][2]+N[2][1]+N[2][3]+N[3][0]+N[3][2])/np.sum(N)
    w1 = 1-2*P-Q
    w2 = 1-2*Q
    #calculation
    d = -0.5*np.log(w1)-0.25*np.log(w2)
    R = (-0.5*np.log(w1)+0.25*np.log(w2))/(-0.5*np.log(w2))
    alpha = R/(R+1)
    beta = 0.5*(1/(R+1))
    p = np.exp(-alpha*d)
    q = np.exp(-beta*d)
    return [p,q,d]

def logLikelihood(p,q,M):
    return np.sum((np.log(K2P(p,q))*M))

def calcParam(p,q):
    Ts = -2.0*np.log(p)*0.125 - 2.0*np.log(p)*0.125 \
         -np.log(q)*(0.25)              #rate of transitions
    Tv = -2*np.log(q)*0.25                              #rate of transversions
    t = Ts + Tv
    R = Ts/Tv                                           #transition/transversion ratio
    alfa = -np.log(p)/t
    beta = 0.5/(0.25*(1.0+R))
    print('alfa: ',alfa,' beta: ',beta,)
    return [R,t]

def postprints(p,q,piA=0.25,piC=0.25,piG=0.25,piT=0.25):
    piR = piA + piG
    piY = piC + piT
    Ts = -2.0*np.log(p)*piA*piG/piR - 2.0*np.log(p)*piC*piT/piY \
         -np.log(q)*(2.0*piA*piG + 2.0*piC*piT)              #rate of transitions
    Tv = -2*np.log(q)*piR*piY                              #rate of transversions
    t = Ts + Tv
    R = Ts/Tv                                           #transition/transversion ratio
    alfa = -np.log(p)/t
    beta = 0.5/(piR*piY*(1.0+R))
    print('alfa: ',alfa,' beta: ',beta,)
    rate = calcRate(piA,piC,piG,piT,alfa,beta)
    print('Rate: ',rate)
 
def readFreqMatrix(file1,file2):
    freq = base_counting.base_count(file1,file2)
    freq = list(map(int,freq))
    return np.reshape(freq,(4,4))
 
def calcRate(piA,piC,piG,piT,alfa,beta):
    piR = piA + piG
    piY = piC + piT
    rate = piA*(alfa*piG/piR+beta*piG + beta*piC + beta*piT) \
        + piG*(alfa*piA/piR+beta*piA + beta*piC + beta*piT) \
        + piC*(beta*piA + beta*piG + alfa*piT/piY+beta*piT) \
        + piT*(beta*piA + beta*piG + alfa*piC/piY+beta*piC)
    #print('rate: ',alfa*0.5 + beta*0.75)
    return rate

tol = np.power(10.0,-12)
EM(tol,readFreqMatrix(sys.argv[1],sys.argv[2]))
