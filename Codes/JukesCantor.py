#!/usr/bin/python3

import numpy as np

# alpha == beta
# p = exp(-alpha*t)

def EM(tolerance):
    #randonmly generated coefficients
    p = np.random.rand(1)
    ProbM = K2P(p)
    Nraw = np.random.multinomial(10000,np.reshape(ProbM,16))    #reshape takes a 1D array as input
    Nraw = np.reshape(Nraw,(4,4))
    
    #initialParam
    p,t = intialParam(Nraw)
    iteration = 0
    convergence = np.inf
    logLold = -np.inf

    while(convergence > tolerance):
        iteration += 1

        ProbM = K2P(p)
        N = Nraw/ProbM

        #E-step
        x_ = N[0][0]+N[1][1]+N[2][2]+N[3][3]        #diagonal of matrix (no change)
        y_ = x_ + N[0][2]+N[2][0]+N[1][3]+N[3][1]   #(no change) + transitions
        s1 = p**2*(x_)*0.25
        s2 = p*(1.0-p)*(y_)*0.125
        s3 = (1.0-p)*N.sum()*0.0625
        
        #M-step
        p = (2*s1+s2)/(2*s1+2*s2+s3)

        #calculation of R,t
        R,t = calcParam(p)
        logLnew = logLikelihood(p,Nraw)
        if(logLold > logLnew):
            print('log Likelihood error')
            break
        
        convergence = np.absolute(logLnew-logLold)
        print(iteration,'log-likelihood= ',logLnew,' R= ',R,' t= ',t)
        logLold = logLnew

def K2P(p):
    v = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            v[i][j] = (p**2*(1 if (i==j) else 0) \
                      + p*(1.0-p)*0.5*(1.0 if ((i%2)==(j%2)) else 0) \
                      + (1.0-p)*0.25)*0.25
    return v

def intialParam(N):
    #auxiliar constants
    P = (N[0][2]+N[1][3]+N[2][0]+N[3][1])/N.sum()
    Q = (N[0][1]+N[0][3]+N[1][0]+N[1][2]+N[2][1]+N[2][3]+N[3][0]+N[3][2])/N.sum()
    w1 = 1-2*P-Q
    w2 = 1-2*Q
    #calculation
    d = -0.5*np.log(w1)-0.25*np.log(w2)
    R = 0.5
    alpha = R/(R+1)
    p = np.exp(-alpha*d)
    return [p,d]

def logLikelihood(p,M):
    return (np.log(K2P(p))*M).sum()

def calcParam(p):
    alpha_t = -np.log(p)
    t = alpha_t + 2*alpha_t
    alpha = alpha_t / t
    R = alpha/(2*alpha)
    return [R,t]

tol = np.power(10.0,-12)
EM(tol)
