#!/usr/bin/python3

import numpy as np
import sys
import csv

import base_counting

def EM_TN(tolerance,N_counts):
    #   p = exp(-alfaY*t)
    #   q = exp(-beta*t)
    #   r = exp(-alfaR*t)

    piA = initialPi(0,N_counts)
    piC = initialPi(1,N_counts)
    piG = initialPi(2,N_counts)
    piT = initialPi(3,N_counts)
    piR = piA+piG
    piY = piC+piT

    # estimate initial values for p,q,r,t using TN distance formula
    p,q,r,t = initialParameters(piA,piC,piG,piT,N_counts)
    #print('Initial estimates: p,q,r',p,' ',q,' ',r)

    iteration = 0
    convergence = np.inf
    logLold = -np.inf

    params = []

    while convergence > tolerance:
        iteration+=1

        #E-step
        S = TN([piA,piC,piG,piT],p,q,r)
        N = N_counts/S
        s1=q*r*(N[0][0]*piA+N[2][2]*piG)
        s2=q*p*(N[1][1]*piC+N[3][3]*piT)
        s3=q*(1.0-r)*((piA**2.0*N[0][0]+piG**2.0*N[2][2])+piA*piG*(N[0][2]+N[2][0]))/piR
        s4=q*(1.0-p)*((piC**2.0*N[1][1]+piT**2.0*N[3][3])+piC*piT*(N[1][3]+N[3][1]))/piY
        s5=calcS5(N,[piA,piC,piG,piT],q)
        s6=calcS(0,2,[piA,piC,piG,piT],p,q,r,N)
        s7=calcS(2,0,[piA,piC,piG,piT],p,q,r,N)
        s8=calcS(1,3,[piA,piC,piG,piT],p,q,r,N)
        s9=calcS(3,1,[piA,piC,piG,piT],p,q,r,N)
        s10=s3
        s11=s4

        #Fisher E-step
        s1_2 = (q*r)**2*(N[0][0]*piA+N[2][2]*piG)
        s2_2 = (q*p)**2*(N[1][1]*piC+N[3][3]*piT)
        s3_2 = q*(1.0-r)**2*(((piA**2.0*N[0][0]+piG**2.0*N[2][2])+piA*piG*(N[0][2]+N[2][0]))/piR)
        s4_2 = q*(1.0-p)**2*(((piC**2.0*N[1][1]+piT**2.0*N[3][3])+piC*piT*(N[1][3]+N[3][1]))/piY)
        s5_2 = calcS5_fisher(N,[piA,piC,piG,piT],q)

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
        params.append([R,t,rho])
        logLnew=logLikelihood([piA,piC,piG,piT],p,q,r,N_counts)
        if (logLold > logLnew):
            print('log Likelihood error')
            return 1
            break
        convergence = np.absolute(logLnew-logLold)
        #print(iteration,'log-likelihood= ',logLnew,' R= ',R,' t=',t,' rho= ',rho)
        logLold=logLnew

    with open('params.csv','w') as f:
        writer = csv.writer(f)
        writer.writerows(params)

    print('fisher information matrix:')
    fisher_info(r,p,q,[s1,s2,s3,s4,s5],[s1_2,s2_2,s3_2,s4_2,s5_2])
    print('t, aR, aY, b, piA, piC, piG, piT')
    return postprints(r,p,q,piA,piC,piG,piT)

def TN(piVector,p,q,r):
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
    return np.sum(np.log(TN(piVector,p,q,r))*M)

def calcS5(N,piVector,q):
    v = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            v[i][j] = (1.0-q)*piVector[j]*piVector[i]*N[i][j]
    return np.sum(v)

def calcS5_fisher(N,piVector,q):
    v = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            v[i][j] = (1.0-q)*(piVector[j]*piVector[i]*N[i][j])**2
    return np.sum(v)

def calcS(i,j,piVector,p,q,r,N):
    y = np.zeros(2)
    z = np.zeros(8)
    alfaK = [r,p]*2
    piK = [piVector[0]+piVector[2],piVector[1]+piVector[3]]
    #calculate X(i,i)
    x = q*alfaK[i]*N[i][i]*piVector[i]
    #calculate Y(-,i)+Y(i,-)
    y[0] = 2.0*(q*(1.0-alfaK[i])*piVector[i]**2/piK[i%2]*N[i][i])
    y[1] = (q*(1.0-alfaK[i])*piVector[i]*piVector[j]/piK[i%2])*(N[i][j]+N[j][i])
    #calculate Z(-,i)+Z(i,-)
    for k in range(4):
        z[k*2]   = (1.0-q)*piVector[i]*piVector[k]*N[i][k]
        z[k*2+1] = (1.0-q)*piVector[k]*piVector[i]*N[k][i]
    return np.sum(y) + np.sum(z) + x

def initialPi(i,N):
    return (np.sum(N[:][i]) + np.sum(N[i][:]))/np.sum(N)

def initialParameters(piA,piC,piG,piT,N):
    piR = piA+piG
    piY = piC+piT
    #auxiliar constants
    k1 = 2.0*piA*piG/piR
    k2 = 2.0*piT*piC/piY
    k3 = 2.0*(piR*piY-piA*piG*piY/piR-piT*piC*piR/piY)
    p1 = (N[0][2]+N[2][0])/np.sum(N)
    p2 = (N[1][3]+N[3][1])/np.sum(N)
    q = (N[0][1]+N[0][3]+N[1][0]+N[1][2]+N[2][1]+N[2][3]+N[3][0]+N[3][2])/np.sum(N)
    w1 = 1.0-p1/k1-q/2.0*piR
    w2 = 1.0-p2/k2-q/2.0*piY
    w3 = 1.0-q/2.0*piR*piY

    d = -k1*np.log(w1)-k2*np.log(w2)-k3*np.log(w3)
    s = -k1*np.log(w1)-k2*np.log(w2)-(k3-2*piR*piY)*np.log(w3)
    v = -2.0*piR*piY*np.log(w3)
    R = s/v
    rho = 1.0

    beta = 0.5/(piR*piY*(1.0+R))
    alfaY = (piR*piY*R-piA*piG-piC*piT)/(2.0*(1.0+R)*(piY*piA*piG*rho+piR*piC*piT))
    alfaR = rho*alfaY

    p = np.exp(-alfaY*d)
    q = np.exp(-beta*d)
    r = np.exp(-alfaR*d)
    return [p,q,r,d]

def calcParameters(piA,piC,piG,piT,p,q,r):
    piR = piA + piG
    piY = piC + piT
    Ts = -2.0*np.log(r)*piA*piG/piR - 2.0*np.log(p)*piC*piT/piY \
        -2.0*np.log(q)*(piA*piG + piC*piT)              #rate of transitions
    Tv = -2.0*np.log(q)*piR*piY                              #rate of transversions
    t = Ts + Tv
    R = Ts/Tv                                           #transition/transversion ratio
    beta = 0.5/(piR*piY*(1.0+R))
    alfaR = -np.log(r)/t
    alfaY = -np.log(p)/t
    rho = alfaR/alfaY
    return [R,t,rho]

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
    beta = -np.log(q)/t
    rate = calcRate(piA,piC,piG,piT,alfaR,alfaY,beta)
    #print('alfaR: ',alfaR,' alfaY: ',alfaY,' beta: ',beta,' piA:',piA,' piC: ',\
    #    piC,' piG: ',piG,' piT ',piT,' Rate: ',rate)
    return [t,alfaR,alfaY,beta,piA,piC,piG,piT]

def readFreqMatrix(file1,file2):
    freq = base_counting.base_count(file1,file2)
    freq = list(map(int,freq))
    return np.reshape(freq,(4,4))

def calcRate(piA,piC,piG,piT,alfaR,alfaY,beta):
    piR = piA + piG
    piY = piC + piT
    rate = piA*(alfaR*piG/piR+beta*piG + beta*piC + beta*piT) \
         + piG*(alfaR*piA/piR+beta*piA + beta*piC + beta*piT) \
         + piC*(beta*piA + beta*piG + alfaY*piT/piY+beta*piT) \
         + piT*(beta*piA + beta*piG + alfaY*piC/piY+beta*piC)
    return rate

def B(r,p,q,s):
    b = np.zeros((3,3),dtype=float)
    b[0][0] = s[0]/(r**2) + s[2]/((1-r)**2)
    b[1][1] = s[2]/(p**2) + s[3]/((1-p)**2)
    b[2][2] = s[0]/(q**2) + s[1]/(q**2) + s[2]/(q**2) + s[3]/(q**2) + \
        s[4]/((1-q)**2)
    return b

def S_2(r,p,q,s,s_2):
    m = np.zeros((3,3),dtype=float)
    m[0][0] = s_2[0]/r**2 - (2*s[0]*s[2])/((1-r)*r) + s_2[2]/(1-r)**2
    m[1][1] = s_2[1]/p**2 - (2*s[1]*s[3])/((1-p)*p) + s_2[3]/(1-p)**2
    m[2][2] = s_2[0]/q**2 + (2*s[0]*s[1])/(q**2) + s_2[1]/q**2 + (2*s[0]*s[2])/q**2 + \
        (2*s[1]*s[2])/q**2 + s_2[2]/q**2 + (2*s[0]*s[3])/q**2 + (2*s[1]*s[3])/q**2 + \
        (2*s[2]*s[3])/q**2 + s_2[3]/q**2 - (2*s[0]*s[4])/((1-q)*q)-(2*s[1]*s[4])/((1-q)*q) - \
        (2*s[2]*s[4])/((1-q)*q) - (2*s[3]*s[4])/((1-q)*q) + s_2[4]/(1-q)**2
    # since S*Tranpose[S] is symmetric
    m[0][1] = m[1][0] = (s[0]/r - s[2]/(1-r)) * (s[1]/p - s[3]/(1-p))
    m[0][2] = m[2][0] = (s[0]/r - s[2]/(1-r)) * (np.sum(s[0:4])/q - s[4]/(1-q))
    m[1][2] = m[2][1] = (s[1]/p - s[3]/(1-p)) * (np.sum(s[0:4])/q - s[4]/(1-q))
    return m

def fisher_info(r,p,q,s,s_2):
    I_y = B(r,p,q,s) - S_2(r,p,q,s,s_2)
    print(I_y)
    print('covariance matrix:')
    cov = np.linalg.inv(I_y)
    print(cov)
    print('B')
    print(B(r,p,q,s))
    print('S*S')
    print(S_2(r,p,q,s,s_2))

def main(args):
    tolerance = np.power(10.0,-15)
    print(np.round(EM_TN(tolerance,readFreqMatrix(args[1],args[2])), decimals = 5))

if __name__ == '__main__':
    main(sys.argv)
