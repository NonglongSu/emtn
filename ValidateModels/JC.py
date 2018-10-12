#!/usr/bin/python3

import numpy as np
import sys

import base_counting

# alpha == beta
# p = exp(-alpha*t)

def EM(tolerance, Nraw):
    #randonmly generated coefficients
    
    #initialParam
    p,t = intialParam(Nraw)
    iteration = 0
    convergence = np.inf
    logLold = -np.inf
    ProbM = JC(p)

    while(convergence > tolerance):
        iteration += 1

        ProbM = JC(p)
        N = Nraw/ProbM

        #E-step
        x_ = N[0][0]+N[1][1]+N[2][2]+N[3][3]        #diagonal of matrix (no change)
        y_ = x_ + N[0][2]+N[2][0]+N[1][3]+N[3][1]   #(no change) + transitions
        s1 = np.power(p,2)*(x_)*0.25
        s2 = p*(1.0-p)*(y_)*0.125
        s3 = (1.0-p)*N.sum()*0.0625
        
        #M-step
        p = (2.0*s1+s2)/(2.0*s1+2.0*s2+s3)
        ## TODO

        #calculation of R,t
        R,t = calcParam(p)
        logLnew = logLikelihood(p,Nraw)
        if(logLold > logLnew):
            print(iteration,'log-L diff= ',logLw-logLold,' R= ',R,' t= ',t)
            print('log Likelihood error')
            print(logLold-logLnew)
            break

        convergence = np.absolute(logLnew-logLold)
        print(iteration,'log-L diff= ',logLnew-logLold,' t= ',t)
        logLold = logLnew

    alpha_t = -np.log(p)
    t = 1.25*alpha_t
    alpha = alpha_t / t
    print('Rate: ',calcRate(alpha))

def JC(p):
    v = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            v[i][j] = (np.power(p,2)*(1.0 if (i==j) else 0) \
                      + p*(1.0-p)*0.5*(1.0 if ((i%2)==(j%2)) else 0) \
                      + (1.0-p)*0.25)*0.25
    return v

def intialParam(N):
    #auxiliar constants
    P = (N[0][2]+N[1][3]+N[2][0]+N[3][1])/N.sum()
    Q = (N[0][1]+N[0][3]+N[1][0]+N[1][2]+N[2][1]+N[2][3]+N[3][0]+N[3][2])/N.sum()
    w1 = 1.0-2.0*P-Q
    w2 = 1.0-2.0*Q
    #calculation
    d = -0.5*np.log(w1)-0.25*np.log(w2)
    R = 0.5
    alpha = R/(R+1.0)
    p = np.exp(-alpha*d)
    return [p,d]

def logLikelihood(p,M):
    return (np.log(JC(p))*M).sum()

def calcParam(p):
    alpha_t = -np.log(p)
    t = 1.25*alpha_t
    alpha = alpha_t / t
    R = alpha/(2.0*alpha)
    return [R,t]

def readFreqMatrix(file1,file2):
    freq = base_counting.base_count(file1,file2)
    freq = list(map(int,freq))
    N = np.reshape(freq,(4,4))
    print(N)
    return N

def calcRate(alpha):
    return 1.25*alpha

tol = np.power(10.0,-12)

EM(tol,readFreqMatrix(sys.argv[1],sys.argv[2]))
