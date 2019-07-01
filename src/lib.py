#!/usr/bin/python3


import sys
import numpy as np

# Generate matrix counts from sequences
def base_count(file1): #,file2):
    # read sequences from dawg
    fasta = open(file1).readlines()
    seq1_raw = fasta[1:fasta.index('\n')]
    seq2_raw = fasta[fasta.index('\n')+2:]
    seq1 = list(filter(lambda char: char != '\n',''.join(seq1_raw)))
    seq2 = list(filter(lambda char: char != '\n',''.join(seq2_raw)))

    # check that length of sequences match
    if(len(seq1) != len(seq2)):
        print("sequences have different length!")
        print(len(seq1),' ',len(seq2))
        print(seq1_raw,'\n',seq2_raw)
        return 1

    # initialize frequency matrix
    freq = {'AA':0, 'AC':0, 'AG':0, 'AT':0, 'CA':0, 'CC':0, 'CG':0, 'CT':0,\
        'GA':0, 'GC':0, 'GG':0, 'GT':0, 'TA':0, 'TC':0, 'TG':0, 'TT':0}

    # count base transitions from seq1 to seq2
    for i in range(len(seq1)):
        freq[seq1[i]+seq2[i]] += 1

    return(list(freq.values()))

def TN(piVector,p,q,r):
    v = np.zeros((4,4))
    alfaK = [r,p]*2
    piK = [piVector[0]+piVector[2],piVector[1]+piVector[3]]*2
    for i in range(4):
        for j in range(4):
            v[i][j] = (alfaK[i]*q*(1.0 if (i==j) else 0) \
                       + q*(1.0-alfaK[i])*piVector[j]*\
                       (1.0 if ((i%2)==(j%2)) else 0)/piK[j] \
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
    return (np.sum(N[:][i]) + np.sum(N[i][:]))/(2*np.sum(N))

def initialParameters(piA,piC,piG,piT,N):
    piR = piA+piG
    piY = piC+piT
    #auxiliar constants
    k1 = 2.0*piA*piG/piR
    k2 = 2.0*piT*piC/piY
    k3 = 2.0*(piR*piY-piA*piG*piY/piR-piT*piC*piR/piY)
    p1 = (N[0][2]+N[2][0])/np.sum(N)
    p2 = (N[1][3]+N[3][1])/np.sum(N)
    q = (N[0][1]+N[0][3]+N[1][0]+N[1][2]+N[2][1]+N[2][3]+N[3][0]+\
        N[3][2])/np.sum(N)
    w1 = 1.0-p1/k1-q/2.0*piR
    w2 = 1.0-p2/k2-q/2.0*piY
    w3 = 1.0-q/2.0*piR*piY
    if(w1<0): w1=np.e
    if(w2<0): w2=np.e
    if(w3<0): w3=np.e

    d = -k1*np.log(w1)-k2*np.log(w2)-k3*np.log(w3)
    s = -k1*np.log(w1)-k2*np.log(w2)-(k3-2*piR*piY)*np.log(w3)
    v = -2.0*piR*piY*np.log(w3)
    R = s/v
    rho = 1.0

    beta = 0.5/(piR*piY*(1.0+R))
    alfaY = (piR*piY*R-piA*piG-piC*piT)/\
        (2.0*(1.0+R)*(piY*piA*piG*rho+piR*piC*piT))
    alfaR = rho*alfaY

    p = np.exp(-alfaY*d)
    q = np.exp(-beta*d)
    r = np.exp(-alfaR*d)
    return [p,q,r,d]

def calcParameters(piA,piC,piG,piT,p,q,r):
    piR = piA + piG
    piY = piC + piT
    Ts = -2.0*np.log(r)*piA*piG/piR - 2.0*np.log(p)*piC*piT/piY \
        -2.0*np.log(q)*(piA*piG + piC*piT)       #rate of transitions
    Tv = -2.0*np.log(q)*piR*piY                  #rate of transversions
    t = Ts + Tv
    R = Ts/Tv                                    #transition/transversion ratio
    beta = 0.5/(piR*piY*(1.0+R))
    alfaR = -np.log(r)/t
    alfaY = -np.log(p)/t
    rho = alfaR/alfaY
    return [R,t,rho]

def postprints(r,p,q,piA,piC,piG,piT):
    piR = piA + piG
    piY = piC + piT
    Ts = -2.0*np.log(r)*piA*piG/piR - 2.0*np.log(p)*piC*piT/piY \
        -np.log(q)*(2.0*piA*piG + 2.0*piC*piT)   #rate of transitions
    Tv = -2*np.log(q)*piR*piY                    #rate of transversions
    t = Ts + Tv
    R = Ts/Tv                                    #transition/transversion ratio
    alfaY = -np.log(p)/t
    alfaR = -np.log(r)/t
    beta = -np.log(q)/t
    rho = alfaR/alfaY
    rate = calcRate(piA,piC,piG,piT,alfaR,alfaY,beta)
    return [t,alfaR,alfaY,beta,piA,piC,piG,piT]

def readFreqMatrix(file1):
    freq = base_count(file1)
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
