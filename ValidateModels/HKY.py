#!/usr/bin/python3

import numpy as np
import sys

import base_counting

def EM(tolerance, NN):
    #   p = exp(-alfaY*t)
    #   q = exp(-beta*t)
    #   r = exp(-alfaR*t)
    
    piA = initialPi(0,NN)
    piC = initialPi(1,NN)
    piG = initialPi(2,NN)
    piT = initialPi(3,NN)
    piR = piA+piG
    piY = piC+piT

    p,q,r,t = initialParameters(piA,piC,piG,piT,np.reshape(NN,(4,4)))
    print('p,q,r',p,' ',q,' ',r)

    iteration = 0
    convergence = np.inf
    logLold = -np.inf

    while convergence > tolerance:
        iteration+=1
        
        #E-step
        S = np.reshape(HKY([piA,piC,piG,piT],p,q,r),16)
        N = NN/S
        s1=q*r*(N[0]*piA+N[10]*piG)
        s2=q*p*(N[5]*piC+N[15]*piT)
        s3=q*(1.0-r)*((piA**2*N[0]+piG**2*N[10])+piA*piG*(N[2]+N[8]))/piR

        s4=q*(1.0-p)*((piC**2*N[5]+piT**2*N[15])+piC*piT*(N[7]+N[13]))/piY
        s5=calcS5(np.reshape(N,(4,4)),[piA,piC,piG,piT],q)
        s6=calcS(0,2,[piA,piC,piG,piT],p,q,r,np.reshape(N,(4,4)))
        s7=calcS(2,0,[piA,piC,piG,piT],p,q,r,np.reshape(N,(4,4)))
        s8=calcS(1,3,[piA,piC,piG,piT],p,q,r,np.reshape(N,(4,4)))
        s9=calcS(3,1,[piA,piC,piG,piT],p,q,r,np.reshape(N,(4,4)))
        s10=s3
        s11=s4
          
        #M-step
        r=s1/(s1+s3)
        p=s2/(s2+s4)
        q=(s1+s2+s3+s4)/(s1+s2+s3+s4+s5)        
        piA = (s6*(-s10+s6+s7))/((s6+s7)*(-s10-s11+s6+s7+s8+s9))
        piC = (s8*(-s11+s8+s9))/((s8+s9)*(-s10-s11+s6+s7+s8+s9))
        piG = (s7*(s10-s6-s7))/((s6+s7)*(s10+s11-s6-s7-s8-s9))
        piT = (1.0-piA-piC-piG)
        piR=piA+piG
        piY=piC+piT
        
        #Calculation of R,t,rho
        R,t,rho = calcParameters(piA,piC,piG,piT,p,q,r)
        logLnew=logLikelihood([piA,piC,piG,piT],p,q,r,NN)
        if (logLold > logLnew):
            print('log Likelihood error')
            break
        convergence = np.absolute(logLnew-logLold)
        print(iteration,'log-likelihood= ',logLnew,' R= ',R,' t =',t,' rho= ',rho)
        logLold=logLnew
    postprints(r,p,q,piA,piC,piG,piT)
  
def HKY(piVector,p,q,r):
    v = np.zeros((4,4))
    alfaK = [r,p]*2
    piK = [piVector[0]+piVector[2],piVector[1]+piVector[3]]*2
    for i in range(4):
        for j in range(4):
            v[i][j] = (alfaK[i]*q*(1.0 if (i==j) else 0) \
                       + q*(1.0-alfaK[i])*piVector[j]*(1.0 if ((i%2)==(j%2)) else 0)\
                                     /piK[j] \
                       + (1.0-q)*piVector[j])*piVector[i]
    return v

def logLikelihood(piVector,p,q,r,M):
    return np.sum((np.log(np.reshape(HKY(piVector,p,q,r),16))*M))

def calcS5(N,piVector,q):
    v = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            v[i][j] = (1.0-q)*piVector[j]*piVector[i]*N[i][j]
    return np.sum(v)

def calcS(i,j,piVector,p,q,r,N):
    y = np.zeros(2)
    z = np.zeros(8)
    alfaK = [r,p]*2
    piK = [piVector[0]+piVector[2],piVector[1]+piVector[3]]
    #calculate X(i,i)
    x = q*alfaK[i]*N[i][i]*piVector[i]
    #calculate Y(-,i)+Y(i,-)
    y[0] = 2*(q*(1.0-alfaK[i])*piVector[i]**2/piK[i%2]*N[i][i])
    y[1] = (q*(1.0-alfaK[i])*piVector[i]*piVector[j]/piK[i%2])*(N[i][j]+N[j][i])
    #calculate Z(-,i)+Z(i,-)
    for k in range(4): 
        z[k*2]   = (1.0-q)*piVector[i]*piVector[k]*N[i][k]
        z[k*2+1] = (1.0-q)*piVector[k]*piVector[i]*N[k][i]
    return np.sum(y) + np.sum(z) + x

def initialPi(i,M):
    N = np.reshape(M,(4,4))
    return (np.sum(N[:][i]) + np.sum(N[i][:]))/np.sum(N)

def initialParameters(piA,piC,piG,piT,N):
    piR = piA+piG
    piY = piC+piT
    #auxiliar constants
    k1 = 2*piA*piG/piR
    k2 = 2*piT*piC/piY
    k3 = 2*(piR*piY-piA*piG*piY/piR-piT*piC*piR/piY)
    p1 = (N[0][2]+N[2][0])/np.sum(N)
    p2 = (N[1][3]+N[3][1])/np.sum(N)
    q = (N[0][1]+N[0][3]+N[1][0]+N[1][2]+N[2][1]+N[2][3]+N[3][0]+N[3][2])/np.sum(N)
    w1 = 1.0-p1/k1-q/2*piR
    w2 = 1.0-p2/k2-q/2*piY
    w3 = 1.0-q/2*piR*piY

    d = -k1*np.log(w1)-k2*np.log(w2)-k3*np.log(w3)
    s = -k1*np.log(w1)-k2*np.log(w2)-(k3-2*piR*piY)*np.log(w3)
    v = -2*piR*piY*np.log(w3)
    R = s/v
    rho = 1.0

    beta = 0.5/(piR*piY*(1.0+R))
    alfaY = (piR*piY*R-piA*piG-piC*piT)/(2*(1.0+R)*(piY*piA*piG*rho+piR*piC*piT))
    alfaR = rho*alfaY

    p = np.exp(-alfaY*d)
    q = np.exp(-beta*d)
    r = np.exp((piR)/(piY)*np.log(p))
    
    return [p,q,r,d]

def calcParameters(piA,piC,piG,piT,p,q,r):
    piR = piA + piG
    piY = piC + piT
    Ts = -2.0*np.log(r)*piA*piG/piR - 2.0*np.log(p)*piC*piT/piY \
        -np.log(q)*(2.0*piA*piG + 2.0*piC*piT)              #rate of transitions
    Tv = -2*np.log(q)*piR*piY                              #rate of transversions
    t = Ts + Tv
    R = Ts/Tv                                           #transition/transversion ratio
    beta = 0.5/(piR*piY*(1.0+R))
    alfaR = -np.log(r)/t
    alfaY = -np.log(p)/t
    rho = alfaR/alfaY
    return [R,t,rho]    

def readFreqMatrix(file1,file2):
    freq = base_counting.base_count(file1,file2)
    freq = list(map(int,freq))
    return freq

def postprints(r,p,q,piA,piC,piG,piT):
     piR = piA + piG
     piY = piC + piT
     Ts = -2.0*np.log(r)*piA*piG/piR - 2.0*np.log(p)*piC*piT/piY \
         -np.log(q)*(2.0*piA*piG + 2.0*piC*piT)              #rate of transitions
     Tv = -2*np.log(q)*piR*piY                              #rate of transversions
     t = Ts + Tv
     R = Ts/Tv                                           #transition/transversion ratio
     alfaY = -np.log(p)/t
     alfaR = -np.log(r)/t
     beta = 0.5/(piR*piY*(1.0+R))
     print('alfaR: ',alfaR,' alfaY: ',alfaY,' beta: ',beta,)
     print(' pi[ACGT]: ',piA,' ',piC,' ',piG,' ',piT)
     rate = calcRate(piA,piC,piG,piT,alfaR,alfaY,beta)
     print(rate)

def calcRate(piA,piC,piG,piT,alfaR,alfaY,beta):
     piR = piA + piG
     piY = piC + piT
     rate = piA*(alfaR*piG/piR+beta*piG + beta*piC + beta*piT) \
          + piG*(alfaR*piA/piR+beta*piA + beta*piC + beta*piT) \
          + piC*(beta*piA + beta*piG + alfaY*piT/piY+beta*piT) \
          + piT*(beta*piA + beta*piG + alfaY*piC/piY+beta*piC)
     return rate


tolerance = np.power(10.0,-12)
EM(tolerance,readFreqMatrix(sys.argv[1],sys.argv[2]))
