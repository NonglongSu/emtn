#!/usr/bin/python3

import numpy as np
import sys

import base_counting

# p = exp(-(4/3)*alpha*t)

def EM(tolerance, Nraw):
    #randonmly generated coefficients
    
    #initialParam
    p,t = intialParam(Nraw)
    iteration = 0
    convergence = np.inf
    logLold = -np.inf
    ProbM = JC(p)
    #print(np.sum(ProbM))
    error = False

    while(convergence > tolerance):
        iteration += 1

        ProbM = JC(p)
        N = Nraw/ProbM

        #E-step
        x = N[0][0]+N[1][1]+N[2][2]+N[3][3]        #diagonal of matrix (no change)
        s1 = p*(x)*0.25
        s2 = (1.0-p)*np.sum(N)*0.0625
        
        #M-step
        p = (s1)/(s1+s2)

        #calculation of R,t
        t = calcParam(p)
        logLnew = logLikelihood(p,Nraw)
        if(logLold > logLnew):
            #print(iteration,'log-L diff= ',logLnew-logLold,' t= ',t)
            print('log Likelihood error. Diff: ', logLold-logLnew)
            error = True
            break

        convergence = np.absolute(logLnew-logLold)
        #print(iteration,'log-L diff= ',convergence,'logL: ',logLnew,' t= ',t)
        logLold = logLnew

    if(not error):
        alpha_t = -np.log(p)
        t = calcParam(p)
        alpha = alpha_t*(3/(4*t))
        print('Rate: ',calcRate(alpha))
        print('t=',t)

def JC(p):
    v = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            v[i][j] = (p*(1.0 if (i==j) else 0) \
                      +0.25*(1.0-p))*0.25
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
    p = np.exp(-(4.0/3.0)*alpha*d)
    return [p,d]

def logLikelihood(p,M):
    return np.sum(np.log(JC(p))*M)

def calcParam(p):
    t = -0.75*np.log(p)
    return t

def readFreqMatrix(file1,file2):
    freq = base_counting.base_count(file1,file2)
    freq = list(map(int,freq))
    N = np.reshape(freq,(4,4))
    return N

def calcRate(alpha):
    return alpha

tol = np.power(10.0,-12)

EM(tol,readFreqMatrix(sys.argv[1],sys.argv[2]))
